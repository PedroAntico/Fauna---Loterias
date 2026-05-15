#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA AVANÇADO DE OTIMIZAÇÃO COMBINATÓRIA - LOTOFÁCIL v2.1
=============================================================
CORREÇÕES CRÍTICAS:
✅ Bug do overlap_score corrigido (clamp)
✅ Threshold de penalidade relaxado (35 em vez de 10)
✅ Penalidades lineares (não quadráticas)
✅ Simulações pré-geradas (não recalcula a cada iteração)
✅ Soft constraints (não hard constraints)
✅ Proxy combinatório rápido para annealing
✅ Validação de premiação apenas no final
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter, defaultdict
from itertools import combinations
import warnings
import os
from datetime import datetime
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

class LotofacilOptimizerV21:
    """
    Otimizador Lotofácil v2.1 - Calibrado e Corrigido
    
    PRINCÍPIOS:
    - Soft constraints (penalidades graduais)
    - Espaço de busca AMPLO (threshold relaxado)
    - Simulações pré-geradas (performance)
    - Proxy combinatório rápido
    - Validação final separada
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """Inicialização com simulações pré-geradas"""
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Parâmetros Lotofácil
        self.TOTAL_NUMBERS = 25
        self.NUMBERS_PER_GAME = 15
        self.MIN_PRIZE = 11
        
        # Conjuntos matemáticos
        self.primes = self._gen_primes(25)
        
        # Constraintes CALIBRADAS (soft)
        self.constraints = self._define_calibrated_constraints()
        
        # Simulações PRÉ-GERADAS
        self.reference_draws = None
        self._pre_generate_simulations()
        
        # Carregar dados
        self._load_historical_data()
        
        print("✅ Otimizador Lotofácil v2.1 inicializado!")
        print(f"📊 {len(self.df)} concursos históricos")
        print(f"🎲 {len(self.reference_draws)} simulações pré-geradas")
        print(f"🎯 Soft constraints calibradas")
    
    def _gen_primes(self, limit):
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def _define_calibrated_constraints(self):
        """
        Constraints CALIBRADAS (soft, não hard)
        
        Diferenças da v2.0:
        - max_per_line: 4 → 5
        - max_per_column: 4 → 5
        - Penalidades LINEARES (não quadráticas)
        - Pesos reduzidos
        """
        return {
            # Blocos consecutivos
            'max_consecutive_block': 5,  # Relaxado (era 4)
            'consecutive_weight': 2.0,    # Linear (era quadrático)
            
            # Linhas do volante (5x5)
            'max_per_line': 5,            # Relaxado (era 4)
            'line_weight': 3.0,           # Reduzido
            
            # Colunas do volante
            'max_per_column': 5,          # Relaxado (era 4)
            'column_weight': 3.0,         # Reduzido
            
            # Redundância de pares
            'max_pair_reuse': 4,          # Máximo de reuso de um par
            'pair_redundancy_weight': 2.0,
            
            # Sobreposição entre bilhetes
            'target_overlap': 8,          # Overlap médio ideal
            'overlap_weight': 2.0,
            
            # Threshold de aceitação para pool inicial
            'acceptance_threshold': 35,   # RELAXADO (era 10!!!)
        }
    
    def _pre_generate_simulations(self, n_draws=5000):
        """
        PRÉ-GERA simulações de sorteios
        
        Usado como referência FIXA para avaliar cobertura
        Evita ruído de Monte Carlo durante otimização
        """
        print("🎲 Pré-gerando simulações de referência...")
        
        self.reference_draws = []
        for _ in range(n_draws):
            drawn = set(np.random.choice(range(1, 26), 15, replace=False))
            self.reference_draws.append(drawn)
        
        print(f"   ✅ {n_draws} sorteios de referência gerados")
    
    def _load_historical_data(self):
        """Carrega dados históricos"""
        print("📂 Carregando dados históricos...")
        
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
            bola_cols = [f'b{i}' for i in range(1, 16)]
            self.df.columns = ['concurso', 'data'] + bola_cols
            self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
            self.dezenas_historicas = self.df[bola_cols].values
            print(f"   ✅ {len(self.df)} concursos carregados")
        except FileNotFoundError:
            print("   ⚠️  Arquivo não encontrado. Gerando dados sintéticos...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_games=3000):
        """Gera dados sintéticos"""
        synthetic_games = []
        for _ in range(n_games):
            weights = np.ones(25) + np.random.normal(0, 0.05, 25)
            weights = np.abs(weights) / np.sum(np.abs(weights))
            game = sorted(np.random.choice(range(1, 26), 15, replace=False, p=weights))
            synthetic_games.append(game)
        
        self.df = pd.DataFrame({
            'concurso': range(1, n_games + 1),
            'data': pd.date_range(start='2003-09-29', periods=n_games, freq='3D')
        })
        
        for i in range(15):
            self.df[f'b{i+1}'] = [g[i] for g in synthetic_games]
        
        bola_cols = [f'b{i}' for i in range(1, 16)]
        self.dezenas_historicas = self.df[bola_cols].values
    
    def _calculate_structural_penalty(self, game):
        """
        Calcula penalidade estrutural LINEAR (soft constraint)
        
        Mudanças da v2.0:
        - Penalidades LINEARES (não quadráticas)
        - Pesos calibrados
        - Sem proibições absolutas
        """
        game_set = set(game)
        penalty = 0.0
        
        c = self.constraints
        
        # 1. Blocos consecutivos (LINEAR)
        d = sorted(game)
        current_block = 1
        max_block = 1
        
        for i in range(len(d) - 1):
            if d[i+1] - d[i] == 1:
                current_block += 1
                max_block = max(max_block, current_block)
            else:
                current_block = 1
        
        if max_block > c['max_consecutive_block']:
            excess = max_block - c['max_consecutive_block']
            penalty += excess * c['consecutive_weight']  # LINEAR
        
        # 2. Linhas do volante (SOFT)
        for row_start in [1, 6, 11, 16, 21]:
            row_numbers = set(range(row_start, row_start + 5))
            overlap = len(game_set & row_numbers)
            
            if overlap > c['max_per_line']:
                excess = overlap - c['max_per_line']
                penalty += excess * c['line_weight']  # LINEAR
        
        # 3. Colunas do volante (SOFT)
        for col_start in range(1, 6):
            col_numbers = set(range(col_start, 26, 5))
            overlap = len(game_set & col_numbers)
            
            if overlap > c['max_per_column']:
                excess = overlap - c['max_per_column']
                penalty += excess * c['column_weight']  # LINEAR
        
        return penalty
    
    def _compute_pair_redundancy(self, pool):
        """
        Calcula redundância de pares
        
        Retorna métricas de quão repetidos estão os pares
        """
        pair_counter = Counter()
        
        for game in pool:
            for pair in combinations(sorted(game), 2):
                pair_counter[pair] += 1
        
        if not pair_counter:
            return 0, 0, 0
        
        counts = list(pair_counter.values())
        avg_redundancy = np.mean(counts)
        max_redundancy = np.max(counts)
        
        # Pares que excedem o limite de reuso
        overused = sum(1 for c in counts if c > self.constraints['max_pair_reuse'])
        
        return avg_redundancy, max_redundancy, overused
    
    def _compute_overlap_stats(self, pool):
        """
        Calcula estatísticas de sobreposição entre bilhetes
        """
        n = len(pool)
        if n < 2:
            return 0, 0, 0
        
        overlaps = []
        for i in range(n):
            for j in range(i+1, n):
                common = len(set(pool[i]) & set(pool[j]))
                overlaps.append(common)
        
        return np.mean(overlaps), np.max(overlaps), np.min(overlaps)
    
    def _fast_prize_proxy(self, pool):
        """
        PROXY RÁPIDO para cobertura de premiação
        
        Usa simulações PRÉ-GERADAS para avaliação consistente
        NÃO gera novas simulações a cada chamada
        """
        if self.reference_draws is None:
            return 0, 0, 0, 0
        
        # Usar subset das simulações para velocidade
        n_eval = min(1000, len(self.reference_draws))
        eval_draws = self.reference_draws[:n_eval]
        
        total_11 = 0
        total_12 = 0
        total_13 = 0
        total_14 = 0
        
        for drawn in eval_draws:
            count_11 = 0
            count_12 = 0
            count_13 = 0
            count_14 = 0
            
            for game in pool:
                hits = len(set(game) & drawn)
                
                if hits == 11:
                    count_11 += 1
                elif hits == 12:
                    count_12 += 1
                elif hits == 13:
                    count_13 += 1
                elif hits >= 14:
                    count_14 += 1
            
            total_11 += count_11
            total_12 += count_12
            total_13 += count_13
            total_14 += count_14
        
        # Médias por sorteio
        avg_11 = total_11 / n_eval
        avg_12 = total_12 / n_eval
        avg_13 = total_13 / n_eval
        avg_14 = total_14 / n_eval
        
        return avg_11, avg_12, avg_13, avg_14
    
    def _compute_pool_fitness(self, pool):
        """
        FITNESS GLOBAL DO POOL - VERSÃO CORRIGIDA
        
        CORREÇÕES:
        ✅ Clamp em todos os scores (0 a max_score)
        ✅ Overlap penalty não pode ser negativo
        ✅ Pesos calibrados
        ✅ Proxy rápido para premiação
        """
        n_games = len(pool)
        
        # 1. PENALIDADES ESTRUTURAIS (25 pontos)
        total_penalty = sum(self._calculate_structural_penalty(g) for g in pool)
        avg_penalty = total_penalty / max(n_games, 1)
        
        # Quanto menor a penalidade, maior o score
        structural_score = max(0, 25 - avg_penalty)
        structural_score = np.clip(structural_score, 0, 25)  # CLAMP
        
        # 2. REDUNDÂNCIA DE PARES (25 pontos)
        avg_red, max_red, overused = self._compute_pair_redundancy(pool)
        
        # Penalidade por redundância
        redundancy_penalty = (avg_red - 1) * self.constraints['pair_redundancy_weight']
        redundancy_penalty += overused * 3
        redundancy_penalty = max(0, redundancy_penalty)  # Não pode ser negativo
        
        redundancy_score = max(0, 25 - redundancy_penalty)
        redundancy_score = np.clip(redundancy_score, 0, 25)  # CLAMP
        
        # 3. SOBREPOSIÇÃO (20 pontos) - BUG CORRIGIDO
        avg_overlap, max_overlap, min_overlap = self._compute_overlap_stats(pool)
        
        # Penalidade por desvio do target
        overlap_deviation = abs(avg_overlap - self.constraints['target_overlap'])
        overlap_penalty = overlap_deviation * self.constraints['overlap_weight']
        overlap_penalty = max(0, overlap_penalty)  # CORRIGIDO: não pode ser negativo
        
        overlap_score = max(0, 20 - overlap_penalty)
        overlap_score = np.clip(overlap_score, 0, 20)  # CLAMP
        
        # 4. COBERTURA DE PREMIAÇÃO - PROXY RÁPIDO (20 pontos)
        avg_11, avg_12, avg_13, avg_14 = self._fast_prize_proxy(pool)
        
        prize_score = (
            avg_11 * 4 +
            avg_12 * 6 +
            avg_13 * 6 +
            avg_14 * 4
        )
        prize_score = np.clip(prize_score, 0, 20)  # CLAMP
        
        # 5. DIVERSIDADE/ENTROPIA (10 pontos)
        all_dezenas = [d for game in pool for d in game]
        dezena_counts = np.bincount(all_dezenas, minlength=26)[1:]
        dezena_probs = dezena_counts / max(np.sum(dezena_counts), 1)
        
        # Evitar log(0)
        dezena_probs = np.where(dezena_probs > 0, dezena_probs, 1e-10)
        pool_entropy = entropy(dezena_probs)
        max_entropy = np.log(25)
        
        entropy_score = (pool_entropy / max_entropy) * 10
        entropy_score = np.clip(entropy_score, 0, 10)  # CLAMP
        
        # FITNESS TOTAL (todos já clamped)
        total_fitness = (
            structural_score +
            redundancy_score +
            overlap_score +
            prize_score +
            entropy_score
        )
        
        return total_fitness, {
            'structural_score': structural_score,
            'redundancy_score': redundancy_score,
            'overlap_score': overlap_score,
            'prize_score': prize_score,
            'entropy_score': entropy_score,
            'avg_structural_penalty': avg_penalty,
            'avg_pair_redundancy': avg_red,
            'max_pair_reuse': max_red,
            'overused_pairs': overused,
            'avg_overlap': avg_overlap,
            'max_overlap': max_overlap,
            'prize_11': avg_11,
            'prize_12': avg_12,
            'prize_13': avg_13,
            'prize_14': avg_14
        }
    
    def _generate_diverse_pool(self, n_games, candidate_pool_size=2000):
        """
        Gera pool inicial DIVERSO com threshold RELAXADO
        
        CORREÇÃO: threshold 35 em vez de 10
        """
        print(f"🎯 Gerando pool inicial diverso (threshold={self.constraints['acceptance_threshold']})...")
        
        candidates = []
        seen = set()
        attempts = 0
        max_attempts = candidate_pool_size * 20  # Mais tentativas
        
        while len(candidates) < candidate_pool_size and attempts < max_attempts:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                penalty = self._calculate_structural_penalty(game)
                
                # Threshold RELAXADO (35 em vez de 10)
                if penalty < self.constraints['acceptance_threshold']:
                    seen.add(game_tuple)
                    candidates.append(game)
            
            attempts += 1
        
        print(f"   ✅ {len(candidates)} candidatos válidos gerados (de {attempts} tentativas)")
        
        if len(candidates) == 0:
            print("   ⚠️  Nenhum candidato válido! Usando geração livre...")
            # Fallback: gerar sem restrições
            while len(candidates) < n_games:
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    candidates.append(game)
            return candidates[:n_games]
        
        # Seleção diversa
        if len(candidates) <= n_games:
            return candidates
        
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        for _ in range(n_games - 1):
            if not remaining:
                break
            
            # Selecionar mais distante dos já escolhidos
            best_idx = 0
            best_min_dist = -1
            
            # Amostrar para eficiência
            sample_size = min(200, len(remaining))
            sample_indices = np.random.choice(len(remaining), sample_size, replace=False)
            
            for idx in sample_indices:
                candidate = remaining[idx]
                min_dist = float('inf')
                
                for sel in selected:
                    common = len(set(candidate) & set(sel))
                    union = len(set(candidate) | set(sel))
                    dist = 1 - common / union if union > 0 else 0
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return selected[:n_games]
    
    def _smart_mutation(self, game):
        """
        Mutação INTELIGENTE que tenta reduzir penalidades
        
        Estratégias:
        1. Trocar dezena problemática
        2. Quebrar blocos consecutivos
        3. Balancear linhas/colunas
        """
        current_penalty = self._calculate_structural_penalty(game)
        
        # Se penalidade já é baixa, mutação aleatória simples
        if current_penalty < 5:
            game = game.copy()
            pos = np.random.randint(0, 15)
            available = [d for d in range(1, 26) if d not in game]
            if available:
                game[pos] = np.random.choice(available)
                game.sort()
            return game
        
        # Tentar reduzir penalidade
        best_game = game.copy()
        best_penalty = current_penalty
        
        # Estratégia 1: Trocar dezenas de blocos consecutivos
        d = sorted(game)
        for i in range(len(d) - 1):
            if d[i+1] - d[i] == 1:
                # Encontrou par consecutivo, tentar quebrar
                test_game = game.copy()
                old_val = d[i]
                available = [x for x in range(1, 26) if x not in test_game]
                
                for new_val in np.random.choice(available, min(5, len(available)), replace=False):
                    test_game[test_game.index(old_val)] = new_val
                    test_game.sort()
                    new_penalty = self._calculate_structural_penalty(test_game)
                    
                    if new_penalty < best_penalty:
                        best_penalty = new_penalty
                        best_game = test_game.copy()
        
        # Estratégia 2: Trocar aleatória se não melhorou
        if best_penalty >= current_penalty:
            best_game = game.copy()
            pos = np.random.randint(0, 15)
            available = [d for d in range(1, 26) if d not in best_game]
            if available:
                best_game[pos] = np.random.choice(available)
                best_game.sort()
        
        return best_game
    
    def _simulated_annealing(self, initial_pool, iterations=1000):
        """
        Simulated Annealing com proxy rápido
        
        NÃO recalcula simulações a cada iteração
        """
        print(f"🔥 Simulated Annealing ({iterations} iterações)...")
        
        current_pool = [list(g) for g in initial_pool]
        current_fitness, current_metrics = self._compute_pool_fitness(current_pool)
        
        best_pool = [g.copy() for g in current_pool]
        best_fitness = current_fitness
        
        fitness_history = [current_fitness]
        
        temp_start = 5.0
        temp_end = 0.05
        
        for iteration in tqdm(range(iterations), desc="Annealing"):
            temp = temp_start * (temp_end / temp_start) ** (iteration / iterations)
            
            # Modificar pool
            new_pool = [g.copy() for g in current_pool]
            
            # Escolher 1-3 jogos para modificar
            n_modifications = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            
            for _ in range(n_modifications):
                idx = np.random.randint(0, len(new_pool))
                
                # Mutação inteligente
                new_pool[idx] = self._smart_mutation(new_pool[idx])
            
            # Avaliar (usa proxy rápido)
            new_fitness, new_metrics = self._compute_pool_fitness(new_pool)
            
            # Metropolis
            delta = new_fitness - current_fitness
            
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_pool = new_pool
                current_fitness = new_fitness
                current_metrics = new_metrics
                
                if current_fitness > best_fitness:
                    best_pool = [g.copy() for g in current_pool]
                    best_fitness = current_fitness
            
            fitness_history.append(current_fitness)
        
        print(f"   ✅ Fitness final: {best_fitness:.2f}")
        return best_pool, fitness_history
    
    def optimize_pool(self, n_games=50, candidate_pool_size=2000, annealing_iterations=1000):
        """
        Pipeline completo de otimização
        """
        print("\n" + "="*60)
        print(f"🎯 OTIMIZANDO POOL DE {n_games} JOGOS (v2.1)")
        print("="*60)
        
        # 1. Pool inicial diverso
        initial_pool = self._generate_diverse_pool(n_games, candidate_pool_size)
        
        print(f"   ✅ Pool inicial: {len(initial_pool)} jogos")
        
        # 2. Simulated Annealing
        optimized_pool, fitness_history = self._simulated_annealing(
            initial_pool, 
            iterations=annealing_iterations
        )
        
        # 3. Avaliação final DETALHADA
        final_fitness, final_metrics = self._compute_pool_fitness(optimized_pool)
        
        print(f"\n📊 MÉTRICAS FINAIS DO POOL (v2.1):")
        print(f"   • Fitness Total: {final_fitness:.2f}/100")
        print(f"   • Score Estrutural: {final_metrics['structural_score']:.2f}/25")
        print(f"   • Score Redundância: {final_metrics['redundancy_score']:.2f}/25")
        print(f"   • Score Sobreposição: {final_metrics['overlap_score']:.2f}/20")
        print(f"   • Score Premiação: {final_metrics['prize_score']:.2f}/20")
        print(f"   • Score Entropia: {final_metrics['entropy_score']:.2f}/10")
        print(f"\n📈 INDICADORES DETALHADOS:")
        print(f"   • Penalidade Estrutural Média: {final_metrics['avg_structural_penalty']:.2f}")
        print(f"   • Redundância Média de Pares: {final_metrics['avg_pair_redundancy']:.2f}")
        print(f"   • Máximo Reuso de Par: {final_metrics['max_pair_reuse']:.0f}x")
        print(f"   • Pares Sobreusados: {final_metrics['overused_pairs']}")
        print(f"   • Overlap Médio: {final_metrics['avg_overlap']:.2f}")
        print(f"   • Overlap Máximo: {final_metrics['max_overlap']:.2f}")
        print(f"   • 11 pts: {final_metrics['prize_11']:.2f}/sorteio")
        print(f"   • 12 pts: {final_metrics['prize_12']:.2f}/sorteio")
        print(f"   • 13 pts: {final_metrics['prize_13']:.2f}/sorteio")
        print(f"   • 14 pts: {final_metrics['prize_14']:.2f}/sorteio")
        
        self.final_metrics = final_metrics
        self.fitness_history = fitness_history
        self.final_pool = optimized_pool
        
        return optimized_pool
    
    def validate_final_pool(self):
        """
        Validação FINAL detalhada com todas as simulações
        
        Executado APENAS no final, não durante otimização
        """
        if not hasattr(self, 'final_pool'):
            print("⚠️  Execute optimize_pool primeiro")
            return
        
        print("\n🔬 VALIDAÇÃO FINAL DETALHADA...")
        
        # Usar TODAS as simulações pré-geradas
        n_eval = len(self.reference_draws)
        
        results = {11: [], 12: [], 13: [], 14: [], 15: []}
        
        for drawn in tqdm(self.reference_draws, desc="Validando"):
            counts = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
            
            for game in self.final_pool:
                hits = len(set(game) & drawn)
                if hits >= 11:
                    counts[hits] += 1
            
            for k in counts:
                results[k].append(counts[k])
        
        print(f"\n📊 RESULTADOS DA VALIDAÇÃO ({n_eval} simulações):")
        for hits in [11, 12, 13, 14, 15]:
            avg = np.mean(results[hits])
            std = np.std(results[hits])
            max_val = np.max(results[hits])
            print(f"   • {hits} pontos: média={avg:.3f} ± {std:.3f} (máx={max_val:.0f})")
        
        return results
    
    def visualize_results(self, output_dir='graficos_lotofacil_v21'):
        """Visualizações dos resultados"""
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        if not hasattr(self, 'final_pool'):
            print("   ⚠️  Execute optimize_pool primeiro")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Evolução do fitness
        ax = axes[0, 0]
        ax.plot(self.fitness_history, color='blue', alpha=0.7, linewidth=1)
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Fitness')
        ax.set_title('Evolução do Fitness')
        ax.grid(True, alpha=0.3)
        
        # 2. Distribuição de penalidades estruturais
        ax = axes[0, 1]
        penalties = [self._calculate_structural_penalty(g) for g in self.final_pool]
        ax.hist(penalties, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x=np.mean(penalties), color='red', linestyle='--', 
                  label=f'Média: {np.mean(penalties):.1f}')
        ax.set_xlabel('Penalidade Estrutural')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Penalidades')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Heatmap de cobertura
        ax = axes[0, 2]
        coverage = np.zeros((5, 5))
        all_dezenas = []
        for game in self.final_pool:
            all_dezenas.extend(game)
        
        for d in all_dezenas:
            row = (d - 1) // 5
            col = (d - 1) % 5
            coverage[row, col] += 1
        
        sns.heatmap(coverage, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[f'C{i+1}' for i in range(5)],
                   yticklabels=[f'L{i+1}' for i in range(5)],
                   ax=ax, cbar_kws={'label': 'Frequência'})
        ax.set_title('Cobertura no Volante 5x5')
        
        # 4. Distribuição de sobreposição
        ax = axes[1, 0]
        overlaps = []
        for i in range(len(self.final_pool)):
            for j in range(i+1, len(self.final_pool)):
                common = len(set(self.final_pool[i]) & set(self.final_pool[j]))
                overlaps.append(common)
        
        ax.hist(overlaps, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=np.mean(overlaps), color='red', linestyle='--', 
                  label=f'Média: {np.mean(overlaps):.1f}')
        ax.set_xlabel('Sobreposição (dezenas em comum)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Sobreposição')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Frequência individual
        ax = axes[1, 1]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        ax.bar(range(1, 26), freq, color='steelblue', edgecolor='black')
        ax.axhline(y=np.mean(freq), color='red', linestyle='--', 
                  label=f'Média: {np.mean(freq):.1f}')
        ax.set_xlabel('Dezena')
        ax.set_ylabel('Frequência')
        ax.set_title('Frequência das Dezenas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Métricas do pool
        ax = axes[1, 2]
        metrics = self.final_metrics
        categories = ['Estrutural', 'Redundância', 'Sobreposição', 'Premiação', 'Entropia']
        values = [
            metrics['structural_score'] / 25 * 100,
            metrics['redundancy_score'] / 25 * 100,
            metrics['overlap_score'] / 20 * 100,
            metrics['prize_score'] / 20 * 100,
            metrics['entropy_score'] / 10 * 100
        ]
        colors = ['green', 'blue', 'orange', 'red', 'purple']
        
        ax.barh(categories, values, color=colors, edgecolor='black')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (%)')
        ax.set_title('Métricas do Pool')
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Análise do Pool Otimizado - Lotofácil v2.1', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analise_pool_v21.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def export_pool(self, games, filename=None):
        """Exporta pool otimizado"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lotofacil_v21_{timestamp}.csv'
        
        clean_games = [[int(x) for x in game] for game in games]
        
        df_games = pd.DataFrame(clean_games, columns=[f'D_{i+1:02d}' for i in range(15)])
        df_games.index = [f'Jogo_{i+1}' for i in range(len(clean_games))]
        
        df_games['Soma'] = [sum(g) for g in clean_games]
        df_games['Pares'] = [sum(1 for n in g if n % 2 == 0) for g in clean_games]
        df_games['Penalidade'] = [self._calculate_structural_penalty(g) for g in clean_games]
        
        df_games.to_csv(filename)
        print(f"\n💾 Pool exportado: {filename}")
        
        return filename


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🎯 SISTEMA AVANÇADO DE OTIMIZAÇÃO - LOTOFÁCIL v2.1")
    print("   Soft Constraints | Proxy Rápido | Simulações Pré-geradas")
    print("="*70)
    
    # Inicializar
    optimizer = LotofacilOptimizerV21()
    
    # Otimizar
    n_games = 50
    optimized_pool = optimizer.optimize_pool(
        n_games=n_games,
        candidate_pool_size=2000,
        annealing_iterations=1000
    )
    
    # Validar (com todas as simulações)
    optimizer.validate_final_pool()
    
    # Visualizar
    optimizer.visualize_results()
    
    # Exportar
    optimizer.export_pool(optimized_pool)
    
    # Mostrar primeiros jogos
    print(f"\n🎯 PRIMEIROS 10 JOGOS DO POOL:")
    print("="*70)
    for i, game in enumerate(optimized_pool[:10], 1):
        clean_game = sorted([int(x) for x in game])
        penalty = optimizer._calculate_structural_penalty(clean_game)
        print(f"Jogo {i:2d}: {clean_game}")
        print(f"        Penalidade={penalty:.1f} | Soma={sum(clean_game)}")
    
    print("\n" + "="*70)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("📁 Resultados salvos em:")
    print("   • graficos_lotofacil_v21/ - Visualizações")
    print("   • lotofacil_v21_*.csv - Jogos exportados")
    print("="*70)


if __name__ == "__main__":
    main()
