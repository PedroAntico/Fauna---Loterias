#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA AVANÇADO DE OTIMIZAÇÃO COMBINATÓRIA - LOTOFÁCIL v2.0
=============================================================
CORREÇÕES CRÍTICAS:
✅ Penalidade pesada para sequências consecutivas
✅ Anti-redundância de pares (pair redundancy)
✅ Penalidade de sobreposição entre bilhetes
✅ Restrições estruturais fortes (linhas/colunas)
✅ Substituição de cobertura absoluta por cobertura útil
✅ Balanceamento estrutural forçado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import hamming, euclidean
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
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

class LotofacilOptimizerV2:
    """
    Otimizador de Cobertura Lotofácil - Versão 2.0
    
    FOCO EM:
    - Cobertura ÚTIL (não absoluta)
    - Anti-redundância de pares
    - Penalidades estruturais fortes
    - Balanceamento de padrões
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """Inicialização"""
        self.historical_csv = historical_csv
        self.df = None
        self.dezenas_historicas = None
        
        # Parâmetros Lotofácil
        self.TOTAL_NUMBERS = 25
        self.NUMBERS_PER_GAME = 15
        self.MIN_PRIZE = 11
        
        # Features simplificadas (foco combinatório)
        self.feature_names = [
            'soma', 'pares', 'impares', 'primos',
            'q1', 'q2', 'q3', 'q4', 'q5',
            'amplitude', 'distancia_media', 'consecutivas'
        ]
        
        # Conjuntos matemáticos
        self.primes = self._gen_primes(25)
        
        # Padrões estruturais proibidos/penalizados
        self.structural_constraints = self._define_structural_constraints()
        
        # Carregar dados
        self._load_historical_data()
        
        print("✅ Otimizador Lotofácil v2.0 inicializado!")
        print(f"📊 {len(self.df)} concursos históricos")
        print(f"🎯 Foco em cobertura ÚTIL e anti-redundância")
    
    def _gen_primes(self, limit):
        return {n for n in range(2, limit+1) 
                if all(n % i != 0 for i in range(2, int(n**0.5)+1))}
    
    def _define_structural_constraints(self):
        """
        Define restrições estruturais FORTES
        para evitar padrões viciados
        """
        return {
            # Blocos consecutivos MÁXIMOS permitidos
            'max_consecutive_block': 4,  # Máximo 4 números consecutivos
            
            # Linhas completas PROIBIDAS (volante 5x5)
            'forbidden_lines': [
                {1,2,3,4,5},
                {6,7,8,9,10},
                {11,12,13,14,15},
                {16,17,18,19,20},
                {21,22,23,24,25}
            ],
            
            # Colunas completas PROIBIDAS
            'forbidden_columns': [
                {1,6,11,16,21},
                {2,7,12,17,22},
                {3,8,13,18,23},
                {4,9,14,19,24},
                {5,10,15,20,25}
            ],
            
            # Máximo de números em uma única linha
            'max_per_line': 4,
            
            # Máximo de números em uma única coluna
            'max_per_column': 4,
            
            # Penalidades
            'consecutive_penalty_weight': 5.0,
            'line_penalty_weight': 10.0,
            'column_penalty_weight': 10.0,
            'pair_redundancy_weight': 3.0,
            'overlap_penalty_weight': 4.0
        }
    
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
        """Gera dados sintéticos balanceados"""
        print("   🔧 Gerando dados sintéticos realistas...")
        
        synthetic_games = []
        for _ in range(n_games):
            # Distribuição mais uniforme
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
        
        print(f"   ✅ {len(self.df)} jogos sintéticos gerados")
    
    def _count_consecutive_blocks(self, game):
        """
        Conta blocos consecutivos e seus tamanhos
        Retorna: (total_consecutivos, max_block_size, num_blocks)
        """
        d = sorted(game)
        blocks = []
        current_block = 1
        
        for i in range(len(d) - 1):
            if d[i+1] - d[i] == 1:
                current_block += 1
            else:
                if current_block >= 2:
                    blocks.append(current_block)
                current_block = 1
        
        if current_block >= 2:
            blocks.append(current_block)
        
        total = sum(blocks) if blocks else 0
        max_block = max(blocks) if blocks else 0
        
        return total, max_block, len(blocks)
    
    def _check_structural_violations(self, game):
        """
        Verifica violações estruturais e retorna penalidade
        
        Penalidades PESADAS para:
        - Blocos consecutivos longos
        - Linhas/colunas muito preenchidas
        - Padrões de volante
        """
        game_set = set(game)
        penalty = 0
        
        constraints = self.structural_constraints
        
        # 1. Blocos consecutivos
        total_cons, max_block, num_blocks = self._count_consecutive_blocks(game)
        
        if max_block > constraints['max_consecutive_block']:
            # Penalidade exponencial para blocos muito longos
            excess = max_block - constraints['max_consecutive_block']
            penalty += excess * excess * constraints['consecutive_penalty_weight']
        
        if total_cons >= 6:
            # Penalidade adicional para muitos consecutivos
            penalty += (total_cons - 5) * constraints['consecutive_penalty_weight']
        
        # 2. Linhas do volante
        for line in constraints['forbidden_lines']:
            overlap = len(game_set & line)
            if overlap > constraints['max_per_line']:
                penalty += (overlap - constraints['max_per_line']) * constraints['line_penalty_weight']
        
        # 3. Colunas do volante
        for col in constraints['forbidden_columns']:
            overlap = len(game_set & col)
            if overlap > constraints['max_per_column']:
                penalty += (overlap - constraints['max_per_column']) * constraints['column_penalty_weight']
        
        return penalty
    
    def _compute_pair_redundancy(self, pool):
        """
        Calcula redundância de pares no pool
        
        Menor redundância = melhor distribuição
        """
        pair_counter = Counter()
        
        for game in pool:
            for pair in combinations(sorted(game), 2):
                pair_counter[pair] += 1
        
        if not pair_counter:
            return 0, 0
        
        counts = list(pair_counter.values())
        avg_redundancy = np.mean(counts)
        max_redundancy = np.max(counts)
        
        # Penalizar pares muito repetidos
        overused = sum(1 for c in counts if c > 3)
        
        return avg_redundancy, overused
    
    def _compute_overlap_matrix(self, pool):
        """
        Calcula matriz de sobreposição entre bilhetes
        """
        n = len(pool)
        overlaps = []
        
        for i in range(n):
            for j in range(i+1, n):
                common = len(set(pool[i]) & set(pool[j]))
                overlaps.append(common)
        
        if not overlaps:
            return 0, 0, 0
        
        return np.mean(overlaps), np.max(overlaps), np.min(overlaps)
    
    def _simulate_prize_coverage(self, pool, n_simulations=5000):
        """
        Simula cobertura de premiação
        """
        results = {11: [], 12: [], 13: [], 14: [], 15: []}
        
        for _ in range(n_simulations):
            drawn = set(np.random.choice(range(1, 26), 15, replace=False))
            
            counts = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
            
            for game in pool:
                hits = len(set(game) & drawn)
                if hits >= 11:
                    counts[hits] += 1
            
            for k in counts:
                results[k].append(counts[k])
        
        return {k: np.mean(v) for k, v in results.items()}
    
    def _compute_pool_fitness(self, pool):
        """
        FITNESS GLOBAL DO POOL - VERSÃO CORRIGIDA
        
        Prioridades:
        1. Minimizar penalidades estruturais
        2. Minimizar redundância de pares
        3. Minimizar sobreposição
        4. Maximizar cobertura de premiação
        5. Manter diversidade
        """
        n_games = len(pool)
        
        # 1. PENALIDADES ESTRUTURAIS (25 pontos - invertido)
        total_structural_penalty = 0
        for game in pool:
            penalty = self._check_structural_violations(game)
            total_structural_penalty += penalty
        
        avg_structural_penalty = total_structural_penalty / n_games
        structural_score = max(0, 25 - avg_structural_penalty * 2)
        
        # 2. REDUNDÂNCIA DE PARES (25 pontos)
        avg_redundancy, overused_pairs = self._compute_pair_redundancy(pool)
        pair_redundancy_penalty = avg_redundancy * self.structural_constraints['pair_redundancy_weight']
        pair_redundancy_penalty += overused_pairs * 2
        redundancy_score = max(0, 25 - pair_redundancy_penalty)
        
        # 3. SOBREPOSIÇÃO (20 pontos)
        avg_overlap, max_overlap, min_overlap = self._compute_overlap_matrix(pool)
        overlap_penalty = (avg_overlap - 8) * self.structural_constraints['overlap_penalty_weight']
        overlap_score = max(0, 20 - overlap_penalty)
        
        # 4. COBERTURA DE PREMIAÇÃO (20 pontos)
        prize_coverage = self._simulate_prize_coverage(pool, n_simulations=2000)
        
        prize_score = (
            prize_coverage[11] * 5 +
            prize_coverage[12] * 7 +
            prize_coverage[13] * 5 +
            prize_coverage[14] * 3
        )
        prize_score = min(prize_score, 20)
        
        # 5. DIVERSIDADE (10 pontos)
        # Distribuição uniforme das dezenas
        all_dezenas = [d for game in pool for d in game]
        dezena_counts = np.bincount(all_dezenas, minlength=26)[1:]
        dezena_probs = dezena_counts / np.sum(dezena_counts)
        pool_entropy = entropy(dezena_probs + 1e-10)
        entropy_score = (pool_entropy / np.log(25)) * 10
        
        # FITNESS TOTAL
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
            'avg_structural_penalty': avg_structural_penalty,
            'avg_pair_redundancy': avg_redundancy,
            'overused_pairs': overused_pairs,
            'avg_overlap': avg_overlap,
            'max_overlap': max_overlap,
            'prize_11': prize_coverage[11],
            'prize_12': prize_coverage[12],
            'prize_13': prize_coverage[13],
            'prize_14': prize_coverage[14]
        }
    
    def _generate_diverse_initial_pool(self, n_games, pool_size=1000):
        """
        Gera pool inicial DIVERSO com restrições estruturais
        """
        print("🎯 Gerando pool inicial diverso...")
        
        candidates = []
        seen = set()
        
        # Gerar candidatos que respeitem restrições estruturais
        attempts = 0
        max_attempts = pool_size * 10
        
        while len(candidates) < pool_size and attempts < max_attempts:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                # Verificar restrições estruturais
                penalty = self._check_structural_violations(game)
                
                # Só aceitar jogos com penalidade baixa
                if penalty < 10:  # Threshold de aceitação
                    seen.add(game_tuple)
                    candidates.append(game)
            
            attempts += 1
        
        print(f"   ✅ {len(candidates)} candidatos válidos gerados")
        
        # Selecionar subconjunto diverso
        if len(candidates) <= n_games:
            return candidates[:n_games]
        
        # Usar seleção gananciosa para diversidade
        selected = [candidates[0]]
        candidates = candidates[1:]
        
        for _ in range(n_games - 1):
            if not candidates:
                break
            
            # Selecionar candidato mais distante dos selecionados
            best_candidate = None
            best_min_distance = -1
            
            for candidate in candidates[:min(100, len(candidates))]:  # Amostra para eficiência
                min_distance = float('inf')
                for sel in selected:
                    # Distância de Jaccard
                    common = len(set(candidate) & set(sel))
                    union = len(set(candidate) | set(sel))
                    distance = 1 - common / union
                    min_distance = min(min_distance, distance)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
        
        return selected[:n_games]
    
    def _simulated_annealing_optimized(self, initial_pool, iterations=1000):
        """
        Simulated Annealing com penalidades estruturais fortes
        """
        print(f"🔥 Simulated Annealing ({iterations} iterações)...")
        
        current_pool = [list(g) for g in initial_pool]
        current_fitness, current_metrics = self._compute_pool_fitness(current_pool)
        
        best_pool = current_pool.copy()
        best_fitness = current_fitness
        
        fitness_history = [current_fitness]
        
        # Parâmetros de annealing
        temp_start = 5.0
        temp_end = 0.05
        
        for iteration in tqdm(range(iterations), desc="Annealing"):
            temp = temp_start * (temp_end / temp_start) ** (iteration / iterations)
            
            # Modificar pool
            new_pool = current_pool.copy()
            
            # Escolher estratégia de modificação
            strategy = np.random.choice(['swap', 'replace', 'shuffle'])
            
            idx = np.random.randint(0, len(new_pool))
            game = new_pool[idx].copy()
            
            if strategy == 'swap':
                # Trocar uma dezena por outra não presente
                old_value = game[np.random.randint(0, 15)]
                available = [d for d in range(1, 26) if d not in game]
                if available:
                    game[game.index(old_value)] = np.random.choice(available)
                    game.sort()
            
            elif strategy == 'replace':
                # Substituir dezena problemática
                # Identificar dezena que causa mais penalidade
                best_replacement = None
                best_penalty_reduction = 0
                
                current_penalty = self._check_structural_violations(game)
                
                for pos in range(15):
                    old_value = game[pos]
                    available = [d for d in range(1, 26) if d not in game]
                    
                    for new_value in np.random.choice(available, min(10, len(available)), replace=False):
                        test_game = game.copy()
                        test_game[pos] = new_value
                        test_game.sort()
                        
                        new_penalty = self._check_structural_violations(test_game)
                        reduction = current_penalty - new_penalty
                        
                        if reduction > best_penalty_reduction:
                            best_penalty_reduction = reduction
                            best_replacement = test_game.copy()
                
                if best_replacement:
                    game = best_replacement
            
            elif strategy == 'shuffle':
                # Embaralhar parte do jogo
                subset_size = np.random.randint(3, 8)
                positions = np.random.choice(15, subset_size, replace=False)
                
                available_pool = [d for d in range(1, 26) if d not in game]
                if len(available_pool) >= subset_size:
                    new_values = np.random.choice(available_pool, subset_size, replace=False)
                    for pos, new_val in zip(positions, new_values):
                        game[pos] = new_val
                    game.sort()
            
            new_pool[idx] = game
            
            # Avaliar novo pool
            new_fitness, new_metrics = self._compute_pool_fitness(new_pool)
            
            # Critério de Metropolis
            delta = new_fitness - current_fitness
            
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_pool = new_pool
                current_fitness = new_fitness
                current_metrics = new_metrics
                
                if current_fitness > best_fitness:
                    best_pool = current_pool.copy()
                    best_fitness = current_fitness
            
            fitness_history.append(current_fitness)
        
        print(f"   ✅ Fitness final: {best_fitness:.2f}")
        return best_pool, fitness_history
    
    def optimize_pool(self, n_games=50, pool_size=1000, annealing_iterations=1000):
        """
        Pipeline completo de otimização
        """
        print("\n" + "="*60)
        print(f"🎯 OTIMIZANDO POOL DE {n_games} JOGOS (v2.0)")
        print("="*60)
        
        # 1. Pool inicial diverso e estruturalmente válido
        initial_pool = self._generate_diverse_initial_pool(n_games, pool_size)
        
        # 2. Simulated Annealing otimizado
        optimized_pool, fitness_history = self._simulated_annealing_optimized(
            initial_pool, 
            iterations=annealing_iterations
        )
        
        # 3. Avaliação final
        final_fitness, final_metrics = self._compute_pool_fitness(optimized_pool)
        
        print(f"\n📊 MÉTRICAS FINAIS DO POOL:")
        print(f"   • Fitness Total: {final_fitness:.2f}/100")
        print(f"   • Score Estrutural: {final_metrics['structural_score']:.2f}/25")
        print(f"   • Score Redundância: {final_metrics['redundancy_score']:.2f}/25")
        print(f"   • Score Sobreposição: {final_metrics['overlap_score']:.2f}/20")
        print(f"   • Score Premiação: {final_metrics['prize_score']:.2f}/20")
        print(f"   • Score Entropia: {final_metrics['entropy_score']:.2f}/10")
        print(f"\n📈 INDICADORES:")
        print(f"   • Penalidade Estrutural Média: {final_metrics['avg_structural_penalty']:.2f}")
        print(f"   • Redundância Média de Pares: {final_metrics['avg_pair_redundancy']:.2f}")
        print(f"   • Pares Sobreusados: {final_metrics['overused_pairs']}")
        print(f"   • Sobreposição Média: {final_metrics['avg_overlap']:.2f}")
        print(f"   • Sobreposição Máxima: {final_metrics['max_overlap']:.2f}")
        print(f"   • 11 pts: {final_metrics['prize_11']:.2f} | 12 pts: {final_metrics['prize_12']:.2f} | 13 pts: {final_metrics['prize_13']:.2f}")
        
        self.final_metrics = final_metrics
        self.fitness_history = fitness_history
        self.final_pool = optimized_pool
        
        return optimized_pool
    
    def visualize_results(self, output_dir='graficos_lotofacil_v2'):
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
        
        # 2. Distribuição de blocos consecutivos
        ax = axes[0, 1]
        max_blocks = []
        for game in self.final_pool:
            _, max_block, _ = self._count_consecutive_blocks(game)
            max_blocks.append(max_block)
        
        block_counts = Counter(max_blocks)
        ax.bar(block_counts.keys(), block_counts.values(), color='steelblue', edgecolor='black')
        ax.axvline(x=4, color='red', linestyle='--', label='Máximo permitido (4)')
        ax.set_xlabel('Tamanho do Maior Bloco Consecutivo')
        ax.set_ylabel('Quantidade de Jogos')
        ax.set_title('Distribuição de Blocos Consecutivos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Heatmap de cobertura (5x5)
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
        ax.set_title('Distribuição de Sobreposição entre Bilhetes')
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
        ax.set_title('Frequência das Dezenas no Pool')
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
        
        ax.barh(categories, values, color=['green', 'blue', 'orange', 'red', 'purple'])
        ax.set_xlim(0, 100)
        ax.set_xlabel('Score (%)')
        ax.set_title('Métricas do Pool')
        for i, v in enumerate(values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        plt.suptitle('Análise do Pool Otimizado - Lotofácil v2.0', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analise_pool_v2.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def export_pool(self, games, filename=None):
        """Exporta pool otimizado"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'lotofacil_otimizado_v2_{timestamp}.csv'
        
        # Converter para tipos Python nativos
        clean_games = []
        for game in games:
            clean_game = [int(x) for x in game]
            clean_games.append(clean_game)
        
        df_games = pd.DataFrame(clean_games, columns=[f'D_{i+1:02d}' for i in range(15)])
        df_games.index = [f'Jogo_{i+1}' for i in range(len(clean_games))]
        
        # Métricas
        df_games['Soma'] = [sum(g) for g in clean_games]
        df_games['Pares'] = [sum(1 for n in g if n % 2 == 0) for g in clean_games]
        df_games['Max_Consecutivos'] = [
            self._count_consecutive_blocks(g)[1] for g in clean_games
        ]
        df_games['Penalidade_Estrutural'] = [
            self._check_structural_violations(g) for g in clean_games
        ]
        
        df_games.to_csv(filename)
        print(f"\n💾 Pool exportado: {filename}")
        
        return filename


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🎯 SISTEMA AVANÇADO DE OTIMIZAÇÃO - LOTOFÁCIL v2.0")
    print("   Anti-redundância | Penalidades Estruturais | Cobertura Útil")
    print("="*70)
    
    # Inicializar
    optimizer = LotofacilOptimizerV2()
    
    # Otimizar
    n_games = 50
    optimized_pool = optimizer.optimize_pool(
        n_games=n_games,
        pool_size=1000,
        annealing_iterations=1000
    )
    
    # Visualizar
    optimizer.visualize_results()
    
    # Exportar
    optimizer.export_pool(optimized_pool)
    
    # Mostrar primeiros jogos
    print(f"\n🎯 PRIMEIROS 10 JOGOS DO POOL OTIMIZADO:")
    print("="*70)
    for i, game in enumerate(optimized_pool[:10], 1):
        clean_game = sorted([int(x) for x in game])
        _, max_block, _ = optimizer._count_consecutive_blocks(clean_game)
        penalty = optimizer._check_structural_violations(clean_game)
        
        # Destacar blocos consecutivos
        game_str = str(clean_game)
        if max_block >= 3:
            game_str += f" ⚠️ Bloco:{max_block}"
        
        print(f"Jogo {i:2d}: {game_str}")
        print(f"        Penalidade={penalty:.1f} | Soma={sum(clean_game)}")
    
    print("\n" + "="*70)
    print("✅ OTIMIZAÇÃO CONCLUÍDA!")
    print("📁 Resultados salvos em:")
    print("   • graficos_lotofacil_v2/ - Visualizações")
    print("   • lotofacil_otimizado_v2_*.csv - Jogos exportados")
    print("="*70)


if __name__ == "__main__":
    main()
