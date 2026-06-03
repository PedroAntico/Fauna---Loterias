#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v40
BUSCA AUTOMÁTICA DE FIXAS + SEMIFIXAS COM INTERVALO

NOVO:
✅ Opção 5: busca automática das melhores combinações de fixas (2, 3, 4 dezenas)
✅ Semifixas agora com intervalo (min_semifixed e max_semifixed)
✅ Gerador respeita o intervalo de semifixas
✅ Análise de frequência histórica das fixas
✅ Walk‑forward condicional
✅ Cobertura condicionada às fixas
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
        """
        Gera um jogo respeitando:
        - fixed: dezenas obrigatórias
        - semifixed: dezenas das quais entre min_semifixed e max_semifixed devem aparecer
        """
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed,
                                      allowed_pares, allowed_moldura, allowed_primos)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed,
                      allowed_pares, allowed_moldura, allowed_primos):
        if fixed is None:
            fixed = []
        if semifixed is None:
            semifixed = []
        
        fixed_set = set(fixed)
        semifixed_set = set(semifixed) - fixed_set
        proibidas = fixed_set | semifixed_set
        todas = set(range(1, 26))
        restantes = list(todas - proibidas)
        
        n_fixas = len(fixed_set)
        
        # Limite superior de semifixas
        if max_semifixed is None:
            max_semi = len(semifixed_set)
        else:
            max_semi = min(max_semifixed, len(semifixed_set))
        
        # Limite inferior
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        
        # Número de semifixas a escolher (aleatório dentro do intervalo)
        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        
        if n_restantes < 0 or n_restantes > len(restantes):
            return None
        
        for _ in range(200):
            # Escolhe semifixas
            if n_semifixed_escolher > 0 and len(semifixed_set) > 0:
                chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set))))
            else:
                chosen_semi = set()
            
            # Escolhe restantes
            if n_restantes > 0:
                chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes))))
            else:
                chosen_rest = set()
            
            game = sorted(fixed_set | chosen_semi | chosen_rest)
            
            if len(game) != 15:
                continue
            
            # Filtros opcionais
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
# OTIMIZADOR DE CARTEIRA COM FIXAS
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
                    fixed=self.fixed,
                    semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed,
                    max_semifixed=self.max_semifixed,
                    allowed_pares=self.allowed_pares,
                    allowed_moldura=self.allowed_moldura,
                    allowed_primos=self.allowed_primos
                )
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

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

    def optimize(self, n_games=5, n_candidates=30000):
        print(f"\n🧩 CARTEIRA COM FIXAS: {n_games} jogos")
        if self.fixed:
            print(f"   Fixas: {self.fixed}")
        if self.semifixed:
            intervalo = f"{self.min_semifixed} a {self.max_semifixed if self.max_semifixed else len(self.semifixed)}"
            print(f"   Semifixas: {self.semifixed} ({intervalo} por jogo)")
        if self.allowed_pares: print(f"   Pares permitidos: {self.allowed_pares}")
        if self.allowed_moldura: print(f"   Moldura permitida: {self.allowed_moldura}")
        if self.allowed_primos: print(f"   Primos permitidos: {self.allowed_primos}")
        
        t0 = time.time()
        pool = self.generate_pool(n_candidates)
        print(f"   Pool gerado: {len(pool)} jogos")
        
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}. Tente relaxar os critérios.")
        
        portfolio = self.select_diverse(pool, n_games)
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
# BUSCA AUTOMÁTICA DAS MELHORES FIXAS (NOVO v40)
# ============================================================
def search_best_fixed(contests, n_fixed=3, top_n=20, min_freq=0.05):
    """
    Testa todas as combinações de n_fixed dezenas e ranqueia por:
    - Frequência conjunta (quantos concursos contêm TODAS as fixas)
    - Quantidade de concursos com 13+ pontos quando as fixas acertam
    """
    print(f"\n🔎 BUSCANDO MELHORES COMBINAÇÕES DE {n_fixed} DEZENAS FIXAS...")
    print(f"   Total de combinações: C(25,{n_fixed}) = {comb(25, n_fixed):,}")
    
    all_combos = list(combinations(range(1, 26), n_fixed))
    results = []
    
    for fixed_tuple in tqdm(all_combos, desc=f"Testando {n_fixed} fixas"):
        fixed_set = set(fixed_tuple)
        total_concursos = len(contests)
        acertos = 0
        concursos_13plus = 0
        concursos_14plus = 0
        
        for c in contests:
            draw_set = set(c['dezenas'])
            if fixed_set.issubset(draw_set):
                acertos += 1
                # Verifica se o concurso tem potencial para 13+ pontos
                # (pelo menos 13 das 15 dezenas estão no sorteio)
                # Com as fixas já certas, precisamos de 10+ das outras 12
                # Simplificamos: conta concursos que fizeram 13+ na história
                # (Isso é uma aproximação — depende da carteira)
        
        freq = acertos / total_concursos
        
        if freq >= min_freq:
            # Agora para esses concursos, verifica quantos tiveram 13+ pontos
            # (usando uma carteira simples de 1 jogo: as fixas + as outras 12 mais sorteadas)
            # Para ranqueamento rápido, usamos uma heurística:
            # conta quantos concursos (dentre os que acertaram as fixas) tiveram
            # pelo menos 13 dezenas iguais a um "jogo médio"
            
            # Cria um jogo de referência: fixas + 12 dezenas mais frequentes do histórico
            freq_counter = Counter()
            for c in contests:
                freq_counter.update(c['dezenas'])
            # Remove as fixas
            for d in fixed_set:
                if d in freq_counter:
                    del freq_counter[d]
            top12 = [d for d, _ in freq_counter.most_common(12)]
            ref_game = sorted(fixed_set | set(top12))
            ref_mask = BITMASK_CACHE.get_mask(ref_game)
            
            for c in contests:
                draw_set = set(c['dezenas'])
                if fixed_set.issubset(draw_set):
                    hits = mask_intersection(ref_mask, BITMASK_CACHE.get_mask(c['dezenas']))
                    if hits >= 13:
                        concursos_13plus += 1
                    if hits >= 14:
                        concursos_14plus += 1
            
            results.append({
                'fixed': fixed_tuple,
                'freq': freq,
                'acertos': acertos,
                '13plus': concursos_13plus,
                '14plus': concursos_14plus,
                'score': freq * (concursos_13plus + 10 * concursos_14plus) / acertos if acertos > 0 else 0
            })
    
    # Ordena por score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Exibe os top_n
    print(f"\n🏆 TOP {top_n} COMBINAÇÕES DE {n_fixed} FIXAS:")
    print(f"{'Rank':<5} {'Fixas':<20} {'Freq':<8} {'Acertos':<8} {'13+':<8} {'14+':<8} {'Score':<10}")
    print("-" * 75)
    for i, res in enumerate(results[:top_n], 1):
        print(f"{i:<5} {str(res['fixed']):<20} {res['freq']:<8.2%} {res['acertos']:<8} "
              f"{res['13plus']:<8} {res['14plus']:<8} {res['score']:<10.4f}")
    
    # Estatísticas
    if results:
        avg_freq = np.mean([r['freq'] for r in results])
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   Média de frequência: {avg_freq:.2%}")
        print(f"   Melhor: {results[0]['fixed']} ({results[0]['freq']:.2%}, "
              f"{results[0]['13plus']}×13+, {results[0]['14plus']}×14+)")
    
    return results

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
        
        # Verifica quantos concursos de teste atendem às condições
        fixed_set = set(fixed)
        semifixed_set = set(semifixed)
        max_semi = max_semifixed if max_semifixed else len(semifixed_set)
        cond_met = 0
        for draw in test_data:
            draw_set = set(draw['dezenas'])
            fixed_hit = fixed_set.issubset(draw_set)
            semifixed_hit = len(semifixed_set & draw_set)
            if fixed_hit and min_semifixed <= semifixed_hit <= max_semi:
                cond_met += 1
        
        opt = PortfolioOptimizer(train_data, fixed, semifixed, min_semifixed, max_semifixed,
                                 allowed_pares, allowed_moldura, allowed_primos)
        portfolio = opt.optimize(n_games, n_candidates=10000)
        bt = opt.backtest(portfolio, test_data)
        
        results.append({
            'window': w,
            'cond_met': cond_met,
            'total_test': len(test_data),
            'lift': bt['lift'],
            'roi': bt['roi'],
            '14pts': bt['hit_distribution'].get(14,0),
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
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v40")
    print("   BUSCA AUTOMÁTICA DE FIXAS + SEMIFIXAS COM INTERVALO")
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
        print("5. Buscar melhores combinações de fixas")
        print("0. Sair")
        op = input("Escolha: ").strip()
        
        if op == '1':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS")
            fixed_str = input("   Dezenas fixas (ex: 2 10 24): ").strip()
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
            portfolio = opt.optimize(5, 30000)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x % 2 == 0)
                pr = sum(1 for x in g if x in PRIMES)
                m = sum(1 for x in g if x in MOLDURA)
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (últimos 200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
        
        elif op == '2':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS PARA ANÁLISE")
            fixed_str = input("   Dezenas fixas (ex: 15 16 20): ").strip()
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
            
            ver_roi = input("\n   Calcular ROI condicionado? (s/n): ").strip().lower()
            if ver_roi == 's':
                portfolio = opt.optimize(5, 30000)
                fixed_set = set(fixed)
                semifixed_set = set(semifixed)
                max_semi = max_semifixed if max_semifixed else len(semifixed_set)
                cond_draws = []
                for draw in contests:
                    draw_set = set(draw['dezenas'])
                    fixed_hit = fixed_set.issubset(draw_set)
                    semifixed_hit = len(semifixed_set & draw_set)
                    if fixed_hit and min_semifixed <= semifixed_hit <= max_semi:
                        cond_draws.append(draw)
                
                if cond_draws:
                    bt = opt.backtest(portfolio, cond_draws)
                    print(f"\n💰 ROI CONDICIONADO ({len(cond_draws)} concursos):")
                    print(f"   Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
                    print(f"   Dist: 11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                          f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)} 15={bt['hit_distribution'].get(15,0)}")
                else:
                    print("   Nenhum concurso atendeu às condições.")
        
        elif op == '3':
            print("\n📝 DEFINIÇÃO DE DEZENAS FIXAS PARA WALK‑FORWARD")
            fixed_str = input("   Dezenas fixas (ex: 2 10 24): ").strip()
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
            fixed_str = input("   Dezenas fixas (ex: 15 16 20): ").strip()
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
            portfolio = opt.optimize(5, 30000)
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
            
            search_best_fixed(contests, n_fixed, top_n)
        
        elif op == '0':
            break
        
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
