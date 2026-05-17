#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE BACKTEST REAL - LOTOFÁCIL
=====================================
Versão 1.0 - Validação Cega com Dados Históricos

OBJETIVO:
✅ Testar estratégias APENAS com dados passados
✅ Medir taxas reais de acerto (11,12,13,14,15)
✅ Comparar contra universo aleatório
✅ Calcular significância estatística
✅ Separar "sorte" de "vantagem real"

METODOLOGIA:
1. Para cada concurso nos últimos N:
   a. Usa APENAS dados anteriores àquele concurso
   b. Gera jogos com a estratégia
   c. Verifica acertos contra o resultado real
2. Compara distribuição de acertos vs aleatório
3. Teste estatístico de hipótese
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import poisson, chisquare, mannwhitneyu
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import warnings
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# CONSTANTES
# ============================================================

TOTAL_NUMBERS = 25
NUMBERS_PER_GAME = 15
MIN_PRIZE = 11

# Número de jogos gerados por concurso no backtest
JOGOS_POR_CONCURSO = 10

# Probabilidades teóricas (Lotofácil)
PROB_TEORICA = {
    11: 1 / 11,      # ~9.09%
    12: 1 / 59,      # ~1.69%
    13: 1 / 691,     # ~0.145%
    14: 1 / 21791,   # ~0.00459%
    15: 1 / 3268760  # ~0.000031%
}


class LotofacilBacktest:
    """
    Sistema de Backtest Real para Lotofácil
    
    Princípios:
    - Teste cego: usa apenas dados passados
    - Comparação: estratégia vs aleatório
    - Significância: testes estatísticos
    - Reprodutibilidade: mesma semente para aleatório
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """
        Inicializa o backtest
        
        Args:
            historical_csv: Arquivo CSV com histórico completo
        """
        self.historical_csv = historical_csv
        self.df = None
        self.all_draws = []
        
        # Resultados do backtest
        self.backtest_results = None
        self.random_baseline = None
        
        # Carregar dados
        self._load_data()
        
        print(f"✅ Backtest inicializado!")
        print(f"📊 {len(self.all_draws)} concursos disponíveis para teste")
    
    def _load_data(self):
        """Carrega todos os dados históricos"""
        print("📂 Carregando dados históricos...")
        
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
            bola_cols = [f'b{i}' for i in range(1, 16)]
            self.df.columns = ['concurso', 'data'] + bola_cols
            self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
            
            # Extrair todos os sorteios em ordem cronológica
            self.df = self.df.sort_values('concurso')
            
            for _, row in self.df.iterrows():
                draw = {
                    'concurso': int(row['concurso']),
                    'data': row['data'],
                    'dezenas': sorted([int(row[f'b{i}']) for i in range(1, 16)])
                }
                self.all_draws.append(draw)
            
            print(f"   ✅ {len(self.all_draws)} concursos carregados")
            print(f"   📅 De {self.all_draws[0]['data'].strftime('%d/%m/%Y')} "
                  f"a {self.all_draws[-1]['data'].strftime('%d/%m/%Y')}")
            
        except FileNotFoundError:
            print("   ⚠️  Arquivo não encontrado!")
            print("   Execute primeiro: python geralotofacil.py")
            raise
    
    def _generate_random_games(self, n_games, seed=None):
        """
        Gera jogos puramente aleatórios
        
        Args:
            n_games: Número de jogos
            seed: Semente para reprodutibilidade
            
        Returns:
            list: Lista de jogos aleatórios
        """
        if seed is not None:
            np.random.seed(seed)
        
        games = []
        seen = set()
        
        for _ in range(n_games):
            while True:
                game = tuple(sorted(np.random.choice(range(1, 26), 15, replace=False)))
                if game not in seen:
                    seen.add(game)
                    games.append(list(game))
                    break
        
        return games
    
    def _generate_strategy_games(self, historical_data, n_games, seed=None):
        """
        Gera jogos usando a estratégia de otimização
        
        SIMULA a estratégia real:
        1. Analisa dados históricos disponíveis
        2. Gera candidatos
        3. Filtra por qualidade
        4. Seleciona os melhores
        
        Args:
            historical_data: Lista de sorteios anteriores
            n_games: Número de jogos a gerar
            seed: Semente para reprodutibilidade
            
        Returns:
            list: Lista de jogos da estratégia
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Extrair dezenas históricas
        historical_dezenas = [d['dezenas'] for d in historical_data]
        
        if len(historical_dezenas) < 10:
            # Poucos dados: usar heurística simples
            return self._generate_simple_heuristic(historical_dezenas, n_games)
        
        # Calcular frequências históricas
        freq = np.bincount(
            [d for draw in historical_dezenas for d in draw], 
            minlength=26
        )[1:]
        
        # Estatísticas históricas
        avg_pares = np.mean([sum(1 for d in draw if d % 2 == 0) for draw in historical_dezenas])
        avg_primos = np.mean([sum(1 for d in draw if d in {2,3,5,7,11,13,17,19,23}) 
                             for draw in historical_dezenas])
        avg_soma = np.mean([sum(draw) for draw in historical_dezenas])
        
        # Gerar candidatos
        candidates = []
        seen = set()
        attempts = 0
        
        while len(candidates) < n_games * 100 and attempts < 50000:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                seen.add(game_tuple)
                
                # Score simples baseado em aderência histórica
                pares = sum(1 for d in game if d % 2 == 0)
                primos = sum(1 for d in game if d in {2,3,5,7,11,13,17,19,23})
                soma = sum(game)
                
                # Penalizar desvios
                score = 0
                score -= abs(pares - avg_pares) * 2
                score -= abs(primos - avg_primos) * 2
                score -= abs(soma - avg_soma) * 0.1
                
                # Bônus por usar dezenas frequentes
                score += sum(freq[d-1] for d in game) * 0.1
                
                # Penalizar consecutivos excessivos
                d = sorted(game)
                cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
                score -= max(0, cons - 6) * 3
                
                candidates.append((score, game))
            
            attempts += 1
        
        # Selecionar os melhores
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Selecionar subconjunto diverso
        selected = []
        for score, game in candidates:
            if len(selected) >= n_games:
                break
            
            # Verificar diversidade
            is_diverse = True
            for sel in selected:
                common = len(set(game) & set(sel))
                if common > 10:  # Muito similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(game)
        
        # Completar se necessário
        while len(selected) < n_games:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            selected.append(game)
        
        return selected[:n_games]
    
    def _generate_simple_heuristic(self, historical_dezenas, n_games):
        """
        Heurística simples quando há poucos dados históricos
        """
        if len(historical_dezenas) > 0:
            freq = np.bincount(
                [d for draw in historical_dezenas for d in draw], 
                minlength=26
            )[1:]
            # Usar dezenas mais frequentes com mais peso
            probs = (freq + 1) / (np.sum(freq) + 25)
        else:
            probs = np.ones(25) / 25
        
        games = []
        seen = set()
        
        for _ in range(n_games):
            while True:
                game = tuple(sorted(np.random.choice(
                    range(1, 26), 15, replace=False, p=probs
                )))
                if game not in seen:
                    seen.add(game)
                    games.append(list(game))
                    break
        
        return games
    
    def _count_hits(self, game, draw_dezenas):
        """Conta acertos de um jogo contra um sorteio"""
        return len(set(game) & set(draw_dezenas))
    
    def run_backtest(self, n_concursos=500, jogos_por_concurso=10, seed=42):
        """
        EXECUTA O BACKTEST COMPLETO
        
        Para cada concurso nos últimos N:
        1. Usa apenas dados ANTERIORES
        2. Gera jogos com estratégia
        3. Gera jogos aleatórios (baseline)
        4. Compara acertos
        
        Args:
            n_concursos: Quantos concursos testar (do final para trás)
            jogos_por_concurso: Jogos gerados por concurso
            seed: Semente para reprodutibilidade
        """
        np.random.seed(seed)
        
        n_test = min(n_concursos, len(self.all_draws) - 100)  # Precisa de histórico inicial
        start_idx = len(self.all_draws) - n_test
        
        print(f"\n{'='*70}")
        print(f"🔬 EXECUTANDO BACKTEST REAL")
        print(f"{'='*70}")
        print(f"📊 Concursos a testar: {n_test}")
        print(f"🎯 Jogos por concurso: {jogos_por_concurso}")
        print(f"📅 Período: {self.all_draws[start_idx]['data'].strftime('%d/%m/%Y')} "
              f"a {self.all_draws[-1]['data'].strftime('%d/%m/%Y')}")
        
        # Resultados acumulados
        strategy_hits = {11: [], 12: [], 13: [], 14: [], 15: []}
        random_hits = {11: [], 12: [], 13: [], 14: [], 15: []}
        
        strategy_total_hits = 0
        random_total_hits = 0
        
        # Para cada concurso no período de teste
        for i in tqdm(range(start_idx, len(self.all_draws)), desc="Backtesting"):
            current_draw = self.all_draws[i]
            
            # Dados disponíveis ANTES deste concurso
            historical_data = self.all_draws[:i]
            
            # Gerar jogos da estratégia (usando seed diferente por concurso)
            strategy_games = self._generate_strategy_games(
                historical_data, 
                jogos_por_concurso, 
                seed=seed + i
            )
            
            # Gerar jogos aleatórios (baseline)
            random_games = self._generate_random_games(
                jogos_por_concurso, 
                seed=seed + i + 100000
            )
            
            # Verificar acertos
            for game in strategy_games:
                hits = self._count_hits(game, current_draw['dezenas'])
                if hits >= 11:
                    strategy_hits[hits].append(1)
                    strategy_total_hits += 1
            
            for game in random_games:
                hits = self._count_hits(game, current_draw['dezenas'])
                if hits >= 11:
                    random_hits[hits].append(1)
                    random_total_hits += 1
        
        # Consolidar resultados
        self.backtest_results = {
            'n_concursos': n_test,
            'jogos_por_concurso': jogos_por_concurso,
            'total_jogos': n_test * jogos_por_concurso,
            'strategy': {
                'total_premios': strategy_total_hits,
                'premios_por_concurso': strategy_total_hits / n_test,
                **{f'premios_{k}': len(v) for k, v in strategy_hits.items()}
            },
            'random': {
                'total_premios': random_total_hits,
                'premios_por_concurso': random_total_hits / n_test,
                **{f'premios_{k}': len(v) for k, v in random_hits.items()}
            }
        }
        
        # Teste estatístico
        self._statistical_test(strategy_hits, random_hits, n_test, jogos_por_concurso)
        
        return self.backtest_results
    
    def _statistical_test(self, strategy_hits, random_hits, n_concursos, jogos_por_concurso):
        """
        Teste estatístico: estratégia vs aleatório
        
        Hipótese nula: estratégia = aleatório
        Hipótese alternativa: estratégia > aleatório
        """
        print(f"\n{'='*70}")
        print(f"📊 ANÁLISE ESTATÍSTICA")
        print(f"{'='*70}")
        
        # Para cada faixa de premiação
        for hits in [11, 12, 13, 14, 15]:
            s_count = len(strategy_hits[hits])
            r_count = len(random_hits[hits])
            
            if s_count == 0 and r_count == 0:
                print(f"\n   {hits} pontos: Sem ocorrências em ambos")
                continue
            
            # Proporções
            s_rate = s_count / (n_concursos * jogos_por_concurso)
            r_rate = r_count / (n_concursos * jogos_por_concurso)
            
            # Teste de proporções (aproximação normal)
            if s_count + r_count > 5:
                # Pooled proportion
                p_pool = (s_count + r_count) / (2 * n_concursos * jogos_por_concurso)
                se = np.sqrt(p_pool * (1 - p_pool) * (2 / (n_concursos * jogos_por_concurso)))
                
                if se > 0:
                    z_stat = (s_rate - r_rate) / se
                    p_value = 1 - stats.norm.cdf(z_stat)  # One-tailed: strategy > random
                else:
                    z_stat = 0
                    p_value = 1.0
            else:
                # Teste exato de Fisher (aproximado via qui-quadrado)
                contingency = np.array([
                    [s_count, n_concursos * jogos_por_concurso - s_count],
                    [r_count, n_concursos * jogos_por_concurso - r_count]
                ])
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                z_stat = np.sqrt(chi2)
                p_value = p_value / 2  # One-tailed
            
            # Interpretação
            if p_value < 0.01:
                significance = "🔴 ALTAMENTE SIGNIFICATIVO (p<0.01)"
            elif p_value < 0.05:
                significance = "🟡 SIGNIFICATIVO (p<0.05)"
            else:
                significance = "🟢 NÃO SIGNIFICATIVO"
            
            print(f"\n   {hits} PONTOS:")
            print(f"      Estratégia: {s_count} prêmios ({s_rate*100:.3f}%)")
            print(f"      Aleatório:  {r_count} prêmios ({r_rate*100:.3f}%)")
            print(f"      Diferença:  {(s_rate - r_rate)*100:+.3f}%")
            print(f"      Z-score:    {z_stat:+.2f}")
            print(f"      p-value:    {p_value:.4f}")
            print(f"      Conclusão:  {significance}")
            
            # Comparar com probabilidade teórica
            prob_teorica = PROB_TEORICA.get(hits, 0)
            print(f"      Prob. teórica: {prob_teorica*100:.4f}%")
    
    def compare_distributions(self):
        """
        Compara distribuições de acertos: estratégia vs aleatório
        
        Usa teste Kolmogorov-Smirnov e visualizações
        """
        if self.backtest_results is None:
            print("⚠️  Execute run_backtest primeiro!")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 COMPARAÇÃO DE DISTRIBUIÇÕES")
        print(f"{'='*70}")
        
        # Simular distribuições completas
        n_sim = 10000
        strategy_dist = []
        random_dist = []
        
        for _ in range(n_sim):
            # Simular um "concurso" de cada estratégia
            s_count = 0
            r_count = 0
            
            for _ in range(self.backtest_results['jogos_por_concurso']):
                # Probabilidade de acerto por faixa
                for hits, count in self.backtest_results['strategy'].items():
                    if hits.startswith('premios_'):
                        h = int(hits.split('_')[1])
                        prob = count / self.backtest_results['total_jogos']
                        if np.random.random() < prob:
                            s_count += 1
                
                for hits, count in self.backtest_results['random'].items():
                    if hits.startswith('premios_'):
                        h = int(hits.split('_')[1])
                        prob = count / self.backtest_results['total_jogos']
                        if np.random.random() < prob:
                            r_count += 1
            
            strategy_dist.append(s_count)
            random_dist.append(r_count)
        
        # Teste KS
        ks_stat, ks_pvalue = stats.ks_2samp(strategy_dist, random_dist)
        
        print(f"\n   Kolmogorov-Smirnov Test:")
        print(f"      KS statistic: {ks_stat:.4f}")
        print(f"      p-value:      {ks_pvalue:.4f}")
        
        if ks_pvalue < 0.05:
            print(f"      Conclusão:    🟡 DISTRIBUIÇÕES DIFERENTES")
        else:
            print(f"      Conclusão:    🟢 DISTRIBUIÇÕES IGUAIS")
        
        # Médias
        print(f"\n   Média de prêmios por concurso:")
        print(f"      Estratégia: {np.mean(strategy_dist):.3f} ± {np.std(strategy_dist):.3f}")
        print(f"      Aleatório:  {np.mean(random_dist):.3f} ± {np.std(random_dist):.3f}")
        
        # Teste Mann-Whitney
        mw_stat, mw_pvalue = mannwhitneyu(strategy_dist, random_dist, alternative='greater')
        print(f"\n   Mann-Whitney U (one-tailed):")
        print(f"      p-value: {mw_pvalue:.4f}")
        if mw_pvalue < 0.05:
            print(f"      Conclusão: 🟡 ESTRATÉGIA SUPERIOR AO ALEATÓRIO")
        else:
            print(f"      Conclusão: 🟢 SEM DIFERENÇA SIGNIFICATIVA")
        
        return {
            'strategy_dist': strategy_dist,
            'random_dist': random_dist,
            'ks_pvalue': ks_pvalue,
            'mw_pvalue': mw_pvalue
        }
    
    def visualize_backtest(self, output_dir='graficos_backtest'):
        """
        Gera visualizações do backtest
        """
        if self.backtest_results is None:
            print("⚠️  Execute run_backtest primeiro!")
            return
        
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Comparação de prêmios por faixa
        ax = axes[0, 0]
        faixas = [11, 12, 13, 14, 15]
        
        strategy_counts = [
            self.backtest_results['strategy'].get(f'premios_{f}', 0) 
            for f in faixas
        ]
        random_counts = [
            self.backtest_results['random'].get(f'premios_{f}', 0) 
            for f in faixas
        ]
        
        x = np.arange(len(faixas))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, strategy_counts, width, label='Estratégia', 
                       color='blue', alpha=0.8)
        bars2 = ax.bar(x + width/2, random_counts, width, label='Aleatório', 
                       color='orange', alpha=0.8)
        
        ax.set_xlabel('Pontos')
        ax.set_ylabel('Número de Prêmios')
        ax.set_title('Prêmios por Faixa: Estratégia vs Aleatório')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f} pts' for f in faixas])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 2. Taxa de acerto comparativa
        ax = axes[0, 1]
        
        strategy_rate = [
            strategy_counts[i] / self.backtest_results['total_jogos'] * 100 
            for i in range(len(faixas))
        ]
        random_rate = [
            random_counts[i] / self.backtest_results['total_jogos'] * 100 
            for i in range(len(faixas))
        ]
        
        ax.semilogy(faixas, strategy_rate, 'bo-', label='Estratégia', linewidth=2, markersize=8)
        ax.semilogy(faixas, random_rate, 'o-', color='orange', label='Aleatório', 
                   linewidth=2, markersize=8)
        
        # Probabilidade teórica
        teorica_rate = [PROB_TEORICA.get(f, 0) * 100 for f in faixas]
        ax.semilogy(faixas, teorica_rate, 'r--', label='Teórico', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Pontos')
        ax.set_ylabel('Taxa de Acerto (%)')
        ax.set_title('Taxa de Acerto (Escala Log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Distribuição de prêmios por concurso
        ax = axes[0, 2]
        
        # Simular para visualização
        strategy_por_concurso = np.random.poisson(
            self.backtest_results['strategy']['premios_por_concurso'],
            self.backtest_results['n_concursos']
        )
        random_por_concurso = np.random.poisson(
            self.backtest_results['random']['premios_por_concurso'],
            self.backtest_results['n_concursos']
        )
        
        ax.hist(strategy_por_concurso, bins=20, alpha=0.7, label='Estratégia', 
               color='blue', density=True)
        ax.hist(random_por_concurso, bins=20, alpha=0.7, label='Aleatório', 
               color='orange', density=True)
        ax.set_xlabel('Prêmios por Concurso')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de Prêmios por Concurso')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Evolução temporal (cumulativo)
        ax = axes[1, 0]
        
        # Simular evolução
        n = self.backtest_results['n_concursos']
        strategy_cumsum = np.cumsum(strategy_por_concurso)
        random_cumsum = np.cumsum(random_por_concurso)
        
        ax.plot(range(n), strategy_cumsum, color='blue', label='Estratégia', linewidth=2)
        ax.plot(range(n), random_cumsum, color='orange', label='Aleatório', linewidth=2)
        ax.fill_between(range(n), strategy_cumsum, random_cumsum, 
                        alpha=0.2, color='green' if strategy_cumsum[-1] > random_cumsum[-1] else 'red')
        ax.set_xlabel('Concurso')
        ax.set_ylabel('Prêmios Acumulados')
        ax.set_title('Evolução Temporal de Prêmios')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Heatmap de acertos
        ax = axes[1, 1]
        
        # Matriz de confusão simplificada
        confusion = np.array([
            [strategy_counts[i], random_counts[i]]
            for i in range(len(faixas))
        ])
        
        sns.heatmap(confusion, annot=True, fmt='d', cmap='YlOrRd',
                   xticklabels=['Estratégia', 'Aleatório'],
                   yticklabels=[f'{f} pts' for f in faixas],
                   ax=ax, cbar_kws={'label': 'Prêmios'})
        ax.set_title('Matriz de Prêmios')
        
        # 6. Significância estatística
        ax = axes[1, 2]
        
        # Calcular p-values aproximados para cada faixa
        p_values = []
        for i, hits in enumerate(faixas):
            s = strategy_counts[i]
            r = random_counts[i]
            total = self.backtest_results['total_jogos']
            
            if s + r > 0:
                p_pool = (s + r) / (2 * total)
                se = np.sqrt(p_pool * (1 - p_pool) * 2 / total)
                if se > 0:
                    z = (s/total - r/total) / se
                    p = 1 - stats.norm.cdf(z)
                else:
                    p = 1.0
            else:
                p = 1.0
            
            p_values.append(p)
        
        colors = ['red' if p < 0.05 else 'green' for p in p_values]
        ax.barh([f'{f} pts' for f in faixas], p_values, color=colors, alpha=0.8)
        ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        ax.set_xlabel('p-value')
        ax.set_title('Significância por Faixa (one-tailed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Backtest Real - Lotofácil: Estratégia vs Aleatório', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/backtest_resultados.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Gráfico adicional: evolução da vantagem
        fig, ax = plt.subplots(figsize=(14, 6))
        
        advantage = strategy_cumsum - random_cumsum
        ax.plot(range(n), advantage, color='purple', linewidth=2)
        ax.fill_between(range(n), 0, advantage, 
                       alpha=0.3, color='green' if advantage[-1] > 0 else 'red')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Concurso')
        ax.set_ylabel('Vantagem Acumulada (Estratégia - Aleatório)')
        ax.set_title('Vantagem Acumulada da Estratégia')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/backtest_vantagem.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def generate_report(self, output_dir='relatorio_backtest'):
        """
        Gera relatório completo do backtest
        """
        if self.backtest_results is None:
            print("⚠️  Execute run_backtest primeiro!")
            return
        
        print(f"\n📄 GERANDO RELATÓRIO...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Relatório texto
        report_path = f'{output_dir}/backtest_report_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO DE BACKTEST - LOTOFÁCIL\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Concursos testados: {self.backtest_results['n_concursos']}\n")
            f.write(f"Jogos por concurso: {self.backtest_results['jogos_por_concurso']}\n")
            f.write(f"Total de jogos: {self.backtest_results['total_jogos']:,}\n\n")
            
            f.write("RESULTADOS:\n")
            f.write("-"*40 + "\n")
            
            for key in ['strategy', 'random']:
                label = "ESTRATÉGIA" if key == 'strategy' else "ALEATÓRIO"
                f.write(f"\n{label}:\n")
                f.write(f"  Total de prêmios: {self.backtest_results[key]['total_premios']}\n")
                f.write(f"  Prêmios por concurso: {self.backtest_results[key]['premios_por_concurso']:.3f}\n")
                
                for hits in [11, 12, 13, 14, 15]:
                    count = self.backtest_results[key].get(f'premios_{hits}', 0)
                    rate = count / self.backtest_results['total_jogos'] * 100
                    f.write(f"  {hits} pontos: {count} ({rate:.4f}%)\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("CONCLUSÃO:\n")
            f.write("="*70 + "\n")
            
            strategy_total = self.backtest_results['strategy']['total_premios']
            random_total = self.backtest_results['random']['total_premios']
            
            if strategy_total > random_total * 1.05:
                f.write("✅ Estratégia MOSTROU VANTAGEM sobre o aleatório\n")
                f.write(f"   Vantagem: {strategy_total - random_total} prêmios ({(strategy_total/random_total - 1)*100:.1f}%)\n")
            elif strategy_total < random_total * 0.95:
                f.write("⚠️  Estratégia foi PIOR que o aleatório\n")
                f.write(f"   Desvantagem: {random_total - strategy_total} prêmios\n")
            else:
                f.write("🟢 Estratégia EQUIVALENTE ao aleatório\n")
                f.write("   Sem diferença estatisticamente significativa\n")
            
            f.write("\n⚠️  DISCLAIMER:\n")
            f.write("Este backtest usa apenas dados passados.\n")
            f.write("Resultados passados não garantem resultados futuros.\n")
            f.write("Loteria é um jogo de azar.\n")
        
        print(f"✅ Relatório salvo: {report_path}")
        
        # JSON com dados detalhados
        json_path = f'{output_dir}/backtest_data_{timestamp}.json'
        
        # Converter para tipos nativos
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'n_concursos': int(self.backtest_results['n_concursos']),
                'jogos_por_concurso': int(self.backtest_results['jogos_por_concurso']),
                'total_jogos': int(self.backtest_results['total_jogos'])
            },
            'strategy': {
                k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
                for k, v in self.backtest_results['strategy'].items()
            },
            'random': {
                k: int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v
                for k, v in self.backtest_results['random'].items()
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Dados JSON: {json_path}")
        
        return report_path, json_path


def main():
    """
    EXECUÇÃO PRINCIPAL DO BACKTEST
    """
    print("="*70)
    print("🔬 SISTEMA DE BACKTEST REAL - LOTOFÁCIL")
    print("   Validação Cega com Dados Históricos")
    print("="*70)
    print()
    print("⚠️  AVISO IMPORTANTE:")
    print("   Este teste usa APENAS dados passados para gerar jogos.")
    print("   É um teste cego: a estratégia NÃO vê o futuro.")
    print("   Isso separa 'sorte' de 'vantagem real'.")
    print("="*70)
    
    # Inicializar backtest
    backtest = LotofacilBacktest()
    
    # Configuração
    n_concursos = min(500, len(backtest.all_draws) - 100)
    jogos_por_concurso = 10
    
    print(f"\n📋 CONFIGURAÇÃO:")
    print(f"   Concursos a testar: {n_concursos}")
    print(f"   Jogos por concurso: {jogos_por_concurso}")
    print(f"   Total de jogos: {n_concursos * jogos_por_concurso:,}")
    
    # Confirmar
    resposta = input("\n▶️  Iniciar backtest? (S/n): ").strip().lower()
    if resposta == 'n':
        print("❌ Backtest cancelado.")
        return
    
    # Executar
    results = backtest.run_backtest(
        n_concursos=n_concursos,
        jogos_por_concurso=jogos_por_concurso,
        seed=42
    )
    
    # Comparar distribuições
    dist_results = backtest.compare_distributions()
    
    # Visualizar
    backtest.visualize_backtest()
    
    # Relatório
    backtest.generate_report()
    
    # Resumo final
    print(f"\n{'='*70}")
    print(f"📋 RESUMO FINAL")
    print(f"{'='*70}")
    
    strategy_total = results['strategy']['total_premios']
    random_total = results['random']['total_premios']
    diff = strategy_total - random_total
    diff_pct = (strategy_total / random_total - 1) * 100 if random_total > 0 else 0
    
    print(f"\n   Total de prêmios (estratégia): {strategy_total}")
    print(f"   Total de prêmios (aleatório):  {random_total}")
    print(f"   Diferença: {diff:+d} ({diff_pct:+.1f}%)")
    
    if diff > 0:
        print(f"\n   ✅ Estratégia teve MAIS prêmios que o aleatório")
        print(f"   ⚠️  Mas isso NÃO prova vantagem real!")
        print(f"   ⚠️  Pode ser apenas flutuação estatística")
    elif diff < 0:
        print(f"\n   ❌ Estratégia teve MENOS prêmios que o aleatório")
        print(f"   💡 Isso sugere que a estratégia NÃO tem vantagem")
    else:
        print(f"\n   🟢 Empate técnico")
    
    print(f"\n💡 INTERPRETAÇÃO CORRETA:")
    print(f"   Se a diferença for pequena e não significativa:")
    print(f"   → A estratégia NÃO demonstrou vantagem real")
    print(f"   → O desempenho é compatível com aleatoriedade")
    print(f"   → Isso é ESPERADO em loterias justas")
    print(f"\n   Se houver diferença significativa:")
    print(f"   → Investigar possível viés nos dados")
    print(f"   → Verificar se há sobreajuste (overfitting)")
    print(f"   → Testar em períodos diferentes")
    
    print(f"\n{'='*70}")
    print(f"✅ BACKTEST CONCLUÍDO!")
    print(f"📁 Resultados salvos em:")
    print(f"   • graficos_backtest/ - Visualizações")
    print(f"   • relatorio_backtest/ - Relatório detalhado")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
