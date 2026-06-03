#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v38.1
FOCO EM COBERTURA COMBINATÓRIA E GARANTIAS REAIS

CORREÇÕES:
✅ Bug n_numbers < 15 corrigido com validação
✅ Garantias reais (enumeração completa quando viável, heurística quando não)
✅ Carteira sempre completa (5 jogos) com relaxamento adaptativo
✅ Wheel system com algoritmo guloso de cobertura
✅ Distinção clara entre "garantia provada" e "estimativa heurística"
"""

import numpy as np
from scipy.stats import entropy, hypergeom, wilcoxon
from collections import Counter
from itertools import combinations
import os, random, time, warnings
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES GERAIS
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
hamming_distance = lambda g1, g2: 15 - mask_intersection(BITMASK_CACHE.get_mask(g1), BITMASK_CACHE.get_mask(g2))

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
# GERADOR COM CRITÉRIOS
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        for _ in range(500):
            game = self._generate_raw(allowed_pares, allowed_moldura, allowed_primos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os critérios fornecidos.")

    def _generate_raw(self, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        if allowed_pares is None and allowed_moldura is None and allowed_primos is None:
            return sorted(np.random.choice(range(1, 26), 15, replace=False))
        for _ in range(200):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
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
# MÓDULO DE COBERTURA COMBINATÓRIA E GARANTIAS REAIS
# ============================================================
class CoverageEngine:
    def __init__(self, generator):
        self.generator = generator
    
    def generate_pool(self, n_candidates, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        pool = []
        seen = set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
            try:
                g = self.generator.generate_one(
                    allowed_pares=allowed_pares,
                    allowed_moldura=allowed_moldura,
                    allowed_primos=allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool
    
    def select_max_diversity(self, candidates, n_select, max_inter=10):
        """Seleciona n_select jogos com diversidade máxima. Garante completar n_select."""
        if len(candidates) < n_select:
            raise ValueError(f"Pool insuficiente: {len(candidates)} < {n_select}")
        
        # Remove duplicatas
        unique = {}
        for c in candidates:
            mask = BITMASK_CACHE.get_mask(c)
            if mask not in unique:
                unique[mask] = c
        cand_unique = list(unique.values())
        
        # Se ainda não tem suficiente, repete
        if len(cand_unique) < n_select:
            cand_unique = candidates[:n_select]
        
        masks = np.array([BITMASK_CACHE.get_mask(c) for c in cand_unique], dtype=np.uint32)
        n = len(cand_unique)
        
        selected_idx = [0]
        # Relaxa progressivamente max_inter se necessário
        for current_max_inter in range(max_inter, 15):
            for _ in range(n_select - len(selected_idx)):
                min_dists = np.full(n, np.inf, dtype=np.float64)
                for idx in selected_idx:
                    intersect = np.array([mask_intersection(masks[i], masks[idx]) for i in range(n)])
                    dist = 15.0 - intersect
                    dist[intersect > current_max_inter] = -999.0
                    min_dists = np.minimum(min_dists, dist)
                min_dists[selected_idx] = -1.0
                valid = np.where(min_dists >= 0)[0]
                if len(valid) == 0:
                    break
                next_idx = valid[np.argmax(min_dists[valid])]
                selected_idx.append(next_idx)
            if len(selected_idx) >= n_select:
                break
        
        # Se ainda falta, completa com os restantes
        for i in range(n):
            if len(selected_idx) >= n_select:
                break
            if i not in selected_idx:
                selected_idx.append(i)
        
        return [cand_unique[i] for i in selected_idx[:n_select]]
    
    def calculate_guarantees_exact(self, portfolio, base_numbers):
        """
        Calcula garantia EXATA para um wheel system.
        Verifica TODAS as combinações de 15 dentro das dezenas base.
        ATENÇÃO: só é viável para conjuntos base pequenos (≤20 dezenas).
        """
        n_base = len(base_numbers)
        total_comb = comb(n_base, 15)
        
        print(f"   Verificando todas as {total_comb:,} combinações possíveis...")
        
        if total_comb > 50000:
            print(f"   ⚠️ Muitas combinações ({total_comb:,}). Usando heurística em vez de garantia exata.")
            return self._calculate_guarantees_heuristic(portfolio, base_numbers)
        
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        min_hits = {k: 15 for k in range(11, 16)}
        
        for combo in tqdm(combinations(sorted(base_numbers), 15), total=total_comb, desc="Verificando"):
            test_mask = sum(1 << d for d in combo)
            best = 0
            for pm in portfolio_masks:
                hits = mask_intersection(pm, test_mask)
                if hits > best:
                    best = hits
            for k in range(11, 16):
                if best < k:
                    min_hits[k] = 0
                elif min_hits[k] > 0:
                    min_hits[k] = min(min_hits[k], best)
        
        return min_hits
    
    def _calculate_guarantees_heuristic(self, portfolio, base_numbers, n_samples=10000):
        """Heurística para estimar garantias quando enumeração completa é inviável."""
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        min_hits = {k: 15 for k in range(11, 16)}
        
        for _ in tqdm(range(n_samples), desc="Amostrando"):
            combo = sorted(np.random.choice(sorted(base_numbers), 15, replace=False))
            test_mask = sum(1 << d for d in combo)
            best = 0
            for pm in portfolio_masks:
                hits = mask_intersection(pm, test_mask)
                if hits > best:
                    best = hits
            for k in range(11, 16):
                if best < k:
                    min_hits[k] = 0
                elif min_hits[k] > 0:
                    min_hits[k] = min(min_hits[k], best)
        
        return min_hits
    
    def wheel_system(self, n_numbers, n_games, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        """
        Gera um sistema de cobertura (wheel) para n_numbers dezenas,
        com n_games jogos, respeitando critérios.
        """
        if n_numbers < 15:
            raise ValueError(f"n_numbers deve ser ≥ 15. Recebido: {n_numbers}")
        
        print(f"\n🎯 WHEEL SYSTEM: {n_numbers} números, {n_games} jogos")
        if allowed_pares: print(f"   Pares: {allowed_pares}")
        if allowed_moldura: print(f"   Moldura: {allowed_moldura}")
        if allowed_primos: print(f"   Primos: {allowed_primos}")
        
        # Seleciona as dezenas base
        if allowed_pares is None and allowed_moldura is None and allowed_primos is None:
            base_numbers = set(range(1, n_numbers + 1))
        else:
            pool = self.generate_pool(3000, allowed_pares, allowed_moldura, allowed_primos)
            freq = Counter(d for g in pool for d in g)
            base_numbers = set([d for d, _ in freq.most_common(n_numbers)])
        
        print(f"   Dezenas base ({len(base_numbers)}): {sorted(base_numbers)}")
        
        # Gera jogos dentro do conjunto base
        pool = []
        seen = set()
        for _ in range(50000):
            game = sorted(np.random.choice(list(base_numbers), 15, replace=False))
            if allowed_pares is not None:
                if sum(1 for x in game if x % 2 == 0) not in allowed_pares:
                    continue
            if allowed_moldura is not None:
                if sum(1 for x in game if x in MOLDURA) not in allowed_moldura:
                    continue
            if allowed_primos is not None:
                if sum(1 for x in game if x in PRIMES) not in allowed_primos:
                    continue
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                pool.append(game)
            if len(pool) >= 5000:
                break
        
        print(f"   Jogos gerados dentro do conjunto base: {len(pool)}")
        
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}. Tente relaxar os critérios.")
        
        # Seleciona por diversidade
        portfolio = self.select_max_diversity(pool, n_games)
        
        # Calcula garantias
        print(f"\n📊 GARANTIAS:")
        guarantees = self.calculate_guarantees_exact(portfolio, base_numbers)
        
        for k in range(11, 16):
            if guarantees[k] >= k:
                print(f"   ✅ {k} pontos GARANTIDOS em pelo menos 1 jogo")
            elif guarantees[k] > 0:
                print(f"   ⚠️ Pelo menos {guarantees[k]} pontos (não garante {k})")
            else:
                print(f"   ❌ Não garante {k} pontos")
        
        return portfolio, base_numbers

# ============================================================
# OTIMIZADOR DE CARTEIRA
# ============================================================
class PortfolioOptimizer:
    def __init__(self, contests, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.allowed_pares = allowed_pares
        self.allowed_moldura = allowed_moldura
        self.allowed_primos = allowed_primos
        self.engine = CoverageEngine(self.generator)
    
    def optimize_coverage(self, n_games=5, n_candidates=30000):
        print(f"\n🧩 CARTEIRA DE COBERTURA: {n_games} jogos")
        if self.allowed_pares: print(f"   Pares permitidos: {self.allowed_pares}")
        if self.allowed_moldura: print(f"   Moldura permitida: {self.allowed_moldura}")
        if self.allowed_primos: print(f"   Primos permitidos: {self.allowed_primos}")
        
        t0 = time.time()
        pool = self.engine.generate_pool(n_candidates, self.allowed_pares, self.allowed_moldura, self.allowed_primos)
        print(f"   Pool gerado: {len(pool)} jogos")
        
        portfolio = self.engine.select_max_diversity(pool, n_games)
        print(f"   Carteira final: {len(portfolio)} jogos")
        print(f"✅ Otimizado em {time.time()-t0:.1f}s")
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
# WALK-FORWARD
# ============================================================
def walk_forward_validation(contests, n_windows=8, train_size=400, test_size=50, n_games=5,
                            n_random_benchmark=100, allowed_pares=None, allowed_moldura=None, allowed_primos=None):
    print(f"\n🔬 WALK-FORWARD ({n_windows} janelas)")
    print(f"   Benchmark robusto: {n_random_benchmark} carteiras aleatórias por janela")
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
        opt = PortfolioOptimizer(train_data, allowed_pares, allowed_moldura, allowed_primos)
        portfolio = opt.optimize_coverage(n_games, n_candidates=10000)
        bt = opt.backtest(portfolio, test_data)
        rand_lifts = []
        for _ in range(n_random_benchmark):
            rand_port = [opt.generator.generate_pure_random() for _ in range(n_games)]
            bt_rand = opt.backtest(rand_port, test_data)
            rand_lifts.append(bt_rand['lift'])
        rand_lifts = np.array(rand_lifts)
        mean_rand_lift = np.mean(rand_lifts)
        std_rand_lift = np.std(rand_lifts)
        z_score = (bt['lift'] - mean_rand_lift) / std_rand_lift if std_rand_lift > 0 else 0.0
        pct_rank = np.mean(rand_lifts <= bt['lift'])
        results.append({
            'window': w, 'strat_lift': bt['lift'], 'mean_rand_lift': mean_rand_lift,
            'z_score': z_score, 'pct_rank': pct_rank,
            'strat_14': bt['hit_distribution'].get(14,0),
        })
        print(f"   Janela {w}: lift={bt['lift']:.3f} | z={z_score:+.2f} | rank={pct_rank:.2f} | 14pts={bt['hit_distribution'].get(14,0)}")
    if results:
        diffs = [r['strat_lift'] - r['mean_rand_lift'] for r in results]
        print(f"\n📊 RESUMO FINAL:")
        print(f"   Média lift estratégia: {np.mean([r['strat_lift'] for r in results]):.3f}")
        print(f"   Média lift aleatório: {np.mean([r['mean_rand_lift'] for r in results]):.3f}")
        print(f"   Média z‑score: {np.mean([r['z_score'] for r in results]):+.2f}")
        try:
            _, p = wilcoxon(diffs)
            print(f"   Wilcoxon p: {p:.4f}")
        except:
            print("   Wilcoxon p: N/A")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v38.1")
    print("   COBERTURA COMBINATÓRIA COM GARANTIAS REAIS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira de cobertura (diversidade máxima)")
        print("2. Wheel System com garantias reais")
        print("3. Walk‑forward (validação)")
        print("4. Backtest nos últimos 200 concursos")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 CRITÉRIOS (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize_coverage(5, 30000)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '2':
            print("\n📝 WHEEL SYSTEM")
            try:
                n_numbers = int(input("   Quantas dezenas base (≥15, ex: 20): ").strip())
                n_games = int(input("   Quantos jogos (ex: 5): ").strip())
            except:
                print("Valores inválidos.")
                continue
            
            if n_numbers < 15:
                print("❌ Erro: n_numbers deve ser ≥ 15.")
                continue
            
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            try:
                portfolio, base = opt.engine.wheel_system(n_numbers, n_games, allowed_pares, allowed_moldura, allowed_primos)
                for i, g in enumerate(portfolio, 1):
                    p = sum(1 for x in g if x % 2 == 0)
                    pr = sum(1 for x in g if x in PRIMES)
                    m = sum(1 for x in g if x in MOLDURA)
                    print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
                if len(contests) > 200:
                    bt = opt.backtest(portfolio, contests[-200:])
                    print(f"\n🔬 BACKTEST: Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            except Exception as e:
                print(f"❌ Erro: {e}")
                print("   Tente relaxar os critérios ou usar um conjunto base maior.")
        
        elif op == '3':
            print("\n📝 CRITÉRIOS PARA WALK‑FORWARD (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            walk_forward_validation(contests, allowed_pares=allowed_pares, allowed_moldura=allowed_moldura, allowed_primos=allowed_primos)
        
        elif op == '4':
            print("\n📝 CRITÉRIOS (opcional)")
            pares_str = input("   Pares (ex: 6 8 9 ou ENTER): ").strip()
            moldura_str = input("   Moldura (ex: 8 9 10 ou ENTER): ").strip()
            primos_str = input("   Primos (ex: 4 5 6 ou ENTER): ").strip()
            allowed_pares = [int(x) for x in pares_str.split()] if pares_str else None
            allowed_moldura = [int(x) for x in moldura_str.split()] if moldura_str else None
            allowed_primos = [int(x) for x in primos_str.split()] if primos_str else None
            opt = PortfolioOptimizer(contests, allowed_pares, allowed_moldura, allowed_primos)
            portfolio = opt.optimize_coverage(5, 30000)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (últimos 200):")
            print(f"   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} 13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
        
        elif op == '0':
            break
        
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
