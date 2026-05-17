#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE BACKTEST AVANÇADO - LOTOFÁCIL v2.0
==============================================
VALIDAÇÃO CIENTÍFICA COMPLETA

MELHORIAS CRÍTICAS:
✅ Baseline aleatória CONTROLADA (mesmas restrições)
✅ Walk-forward validation (múltiplas janelas)
✅ Monte Carlo massivo (100+ seeds)
✅ Intervalos de confiança (IC95%)
✅ ROI esperado (valor financeiro real)
✅ Teste de hipótese rigoroso
✅ Correção do bug 'premios_por_concurso'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, poisson, mannwhitneyu
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import warnings
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 150

# ============================================================
# CONSTANTES
# ============================================================

TOTAL_NUMBERS = 25
NUMBERS_PER_GAME = 15
MIN_PRIZE = 11

# Probabilidades teóricas (Lotofácil)
PROB_TEORICA = {
    11: 1 / 11,
    12: 1 / 59,
    13: 1 / 691,
    14: 1 / 21791,
    15: 1 / 3268760
}

# Valores de prêmio (estimativas médias - Lotofácil)
VALOR_PREMIO = {
    11: 5.00,       # R$ 5,00
    12: 10.00,      # R$ 10,00
    13: 30.00,      # R$ 30,00
    14: 1500.00,    # R$ 1.500,00
    15: 1500000.00  # R$ 1.500.000,00
}

CUSTO_APOSTA = 3.00  # R$ 3,00 por aposta


class AdvancedBacktest:
    """
    Backtest Avançado com Baseline Controlada e Walk-Forward
    
    PRINCÍPIOS:
    - Baseline com MESMAS restrições estruturais
    - Múltiplas janelas de validação
    - Monte Carlo para significância real
    - ROI financeiro (não apenas contagem)
    """
    
    def __init__(self, historical_csv='resultados_lotofacil.csv'):
        """Inicialização"""
        self.historical_csv = historical_csv
        self.df = None
        self.all_draws = []
        
        # Resultados
        self.backtest_results = None
        self.monte_carlo_results = []
        self.walk_forward_results = []
        
        # Carregar dados
        self._load_data()
        
        print(f"✅ Backtest Avançado v2.0 inicializado!")
        print(f"📊 {len(self.all_draws)} concursos disponíveis")
    
    def _load_data(self):
        """Carrega dados históricos"""
        print("📂 Carregando dados...")
        
        try:
            self.df = pd.read_csv(self.historical_csv, sep=';', encoding='utf-8')
            bola_cols = [f'b{i}' for i in range(1, 16)]
            self.df.columns = ['concurso', 'data'] + bola_cols
            self.df['data'] = pd.to_datetime(self.df['data'], format='%d/%m/%Y', errors='coerce')
            self.df = self.df.sort_values('concurso')
            
            for _, row in self.df.iterrows():
                self.all_draws.append({
                    'concurso': int(row['concurso']),
                    'data': row['data'],
                    'dezenas': sorted([int(row[f'b{i}']) for i in range(1, 16)])
                })
            
            print(f"   ✅ {len(self.all_draws)} concursos carregados")
        except FileNotFoundError:
            print("   ⚠️  Arquivo não encontrado!")
            raise
    
    def _get_historical_stats(self, draws):
        """
        Extrai estatísticas de um conjunto de sorteios
        
        Returns:
            dict: Estatísticas agregadas
        """
        if not draws:
            return None
        
        dezenas_list = [d['dezenas'] for d in draws]
        
        pares_list = [sum(1 for x in d if x % 2 == 0) for d in dezenas_list]
        primos_list = [sum(1 for x in d if x in {2,3,5,7,11,13,17,19,23}) for d in dezenas_list]
        soma_list = [sum(d) for d in dezenas_list]
        
        return {
            'avg_pares': np.mean(pares_list),
            'std_pares': np.std(pares_list),
            'min_pares': np.min(pares_list),
            'max_pares': np.max(pares_list),
            'avg_primos': np.mean(primos_list),
            'std_primos': np.std(primos_list),
            'avg_soma': np.mean(soma_list),
            'std_soma': np.std(soma_list),
            'min_soma': np.min(soma_list),
            'max_soma': np.max(soma_list),
        }
    
    def _generate_strategy_games(self, historical_data, n_games, seed=None):
        """
        Estratégia de otimização (usando apenas dados passados)
        """
        if seed is not None:
            np.random.seed(seed)
        
        historical_dezenas = [d['dezenas'] for d in historical_data]
        
        if len(historical_dezenas) < 10:
            return self._generate_random_constrained(historical_dezenas, n_games, seed)
        
        # Frequências
        freq = np.bincount(
            [d for draw in historical_dezenas for d in draw], 
            minlength=26
        )[1:]
        
        # Estatísticas
        stats_hist = self._get_historical_stats(historical_data)
        avg_pares = stats_hist['avg_pares']
        avg_primos = stats_hist['avg_primos']
        avg_soma = stats_hist['avg_soma']
        
        # Gerar candidatos com score
        candidates = []
        seen = set()
        attempts = 0
        
        while len(candidates) < n_games * 100 and attempts < 50000:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                seen.add(game_tuple)
                
                pares = sum(1 for d in game if d % 2 == 0)
                primos = sum(1 for d in game if d in {2,3,5,7,11,13,17,19,23})
                soma = sum(game)
                cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
                
                # Score multiobjetivo
                score = 0
                score -= abs(pares - avg_pares) * 2
                score -= abs(primos - avg_primos) * 2
                score -= abs(soma - avg_soma) * 0.1
                score += sum(freq[d-1] for d in game) * 0.1
                score -= max(0, cons - 6) * 3
                
                candidates.append((score, game))
            
            attempts += 1
        
        # Selecionar melhores com diversidade
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        for score, game in candidates:
            if len(selected) >= n_games:
                break
            
            is_diverse = True
            for sel in selected:
                common = len(set(game) & set(sel))
                if common > 10:
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(game)
        
        while len(selected) < n_games:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            selected.append(game)
        
        return selected[:n_games]
    
    def _generate_random_constrained(self, historical_dezenas, n_games, seed=None):
        """
        BASELINE ALEATÓRIA CONTROLADA
        
        Gera jogos aleatórios com AS MESMAS RESTRIÇÕES
        que a estratégia usa (faixa de pares, soma, etc.)
        
        Isso torna o teste JUSTO:
        - Ambas estratégias removem jogos "grotescos"
        - A diferença (se houver) é estrutural, não trivial
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Se temos dados históricos, usar estatísticas deles
        if len(historical_dezenas) > 0:
            pares_hist = [sum(1 for x in d if x % 2 == 0) for d in historical_dezenas]
            soma_hist = [sum(d) for d in historical_dezenas]
            cons_hist = [sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1) for d in historical_dezenas]
            
            pares_min = max(4, int(np.percentile(pares_hist, 10)))
            pares_max = min(11, int(np.percentile(pares_hist, 90)))
            soma_min = int(np.percentile(soma_hist, 10))
            soma_max = int(np.percentile(soma_hist, 90))
            cons_max = min(8, int(np.percentile(cons_hist, 90)))
        else:
            # Valores padrão razoáveis
            pares_min, pares_max = 6, 9
            soma_min, soma_max = 170, 220
            cons_max = 8
        
        games = []
        seen = set()
        attempts = 0
        max_attempts = n_games * 100
        
        while len(games) < n_games and attempts < max_attempts:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            game_tuple = tuple(game)
            
            if game_tuple not in seen:
                pares = sum(1 for x in game if x % 2 == 0)
                soma = sum(game)
                cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i]==1)
                
                # MESMAS RESTRIÇÕES DA ESTRATÉGIA
                if (pares_min <= pares <= pares_max and
                    soma_min <= soma <= soma_max and
                    cons <= cons_max):
                    
                    seen.add(game_tuple)
                    games.append(game)
            
            attempts += 1
        
        # Fallback se não gerou suficiente
        while len(games) < n_games:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                games.append(game)
        
        return games[:n_games]
    
    def _count_hits(self, game, draw_dezenas):
        """Conta acertos"""
        return len(set(game) & set(draw_dezenas))
    
    def _calculate_roi(self, hits_dict, n_jogos, custo_aposta=CUSTO_APOSTA):
        """
        Calcula ROI financeiro
        
        Args:
            hits_dict: Dicionário com contagem de prêmios por faixa
            n_jogos: Total de jogos realizados
            custo_aposta: Custo unitário da aposta
            
        Returns:
            dict: Métricas financeiras
        """
        receita = 0
        for hits, count in hits_dict.items():
            if isinstance(hits, int) and hits in VALOR_PREMIO:
                receita += count * VALOR_PREMIO[hits]
        
        custo_total = n_jogos * custo_aposta
        lucro = receita - custo_total
        roi = (lucro / custo_total) * 100 if custo_total > 0 else 0
        
        return {
            'receita': receita,
            'custo': custo_total,
            'lucro': lucro,
            'roi_pct': roi
        }
    
    def run_single_backtest(self, n_concursos, jogos_por_concurso, seed):
        """
        Executa UM backtest com seed específica
        
        Returns:
            dict: Resultados do backtest
        """
        np.random.seed(seed)
        
        n_test = min(n_concursos, len(self.all_draws) - 100)
        start_idx = len(self.all_draws) - n_test
        
        # Acumuladores
        strategy_hits = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        random_hits = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
        
        # Para cada concurso
        for i in range(start_idx, len(self.all_draws)):
            current_draw = self.all_draws[i]
            historical_data = self.all_draws[:i]
            
            # Gerar jogos
            strategy_games = self._generate_strategy_games(
                historical_data, jogos_por_concurso, seed=seed + i
            )
            
            random_games = self._generate_random_constrained(
                [d['dezenas'] for d in historical_data], 
                jogos_por_concurso, 
                seed=seed + i + 100000
            )
            
            # Verificar acertos
            for game in strategy_games:
                hits = self._count_hits(game, current_draw['dezenas'])
                if hits >= 11:
                    strategy_hits[hits] += 1
            
            for game in random_games:
                hits = self._count_hits(game, current_draw['dezenas'])
                if hits >= 11:
                    random_hits[hits] += 1
        
        total_jogos = n_test * jogos_por_concurso
        
        # Calcular ROI
        strategy_roi = self._calculate_roi(strategy_hits, total_jogos)
        random_roi = self._calculate_roi(random_hits, total_jogos)
        
        return {
            'seed': seed,
            'n_concursos': n_test,
            'jogos_por_concurso': jogos_por_concurso,
            'total_jogos': total_jogos,
            'strategy': {
                'hits': strategy_hits.copy(),
                'total_premios': sum(strategy_hits.values()),
                'roi': strategy_roi
            },
            'random': {
                'hits': random_hits.copy(),
                'total_premios': sum(random_hits.values()),
                'roi': random_roi
            }
        }
    
    def run_monte_carlo_backtest(self, n_concursos=300, jogos_por_concurso=10, n_seeds=100):
        """
        MONTE CARLO MASSIVO
        
        Executa backtest com MÚLTIPLAS seeds
        para obter distribuição real da vantagem
        """
        print(f"\n{'='*70}")
        print(f"🎲 MONTE CARLO BACKTEST")
        print(f"{'='*70}")
        print(f"📊 Configuração:")
        print(f"   Concursos por teste: {n_concursos}")
        print(f"   Jogos por concurso: {jogos_por_concurso}")
        print(f"   Seeds (repetições): {n_seeds}")
        print(f"   Total de simulações: {n_seeds * n_concursos * jogos_por_concurso:,}")
        
        self.monte_carlo_results = []
        
        for seed in tqdm(range(n_seeds), desc="Monte Carlo"):
            result = self.run_single_backtest(n_concursos, jogos_por_concurso, seed)
            self.monte_carlo_results.append(result)
        
        # Consolidar
        self._analyze_monte_carlo_results()
        
        return self.monte_carlo_results
    
    def _analyze_monte_carlo_results(self):
        """
        Analisa resultados do Monte Carlo
        
        Calcula:
        - Vantagem média por faixa
        - Intervalos de confiança
        - Significância estatística
        - Distribuição de ROI
        """
        if not self.monte_carlo_results:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 ANÁLISE MONTE CARLO ({len(self.monte_carlo_results)} simulações)")
        print(f"{'='*70}")
        
        # Extrair diferenças
        for hits in [11, 12, 13, 14, 15]:
            strategy_counts = []
            random_counts = []
            differences = []
            
            for result in self.monte_carlo_results:
                s = result['strategy']['hits'].get(hits, 0)
                r = result['random']['hits'].get(hits, 0)
                strategy_counts.append(s)
                random_counts.append(r)
                differences.append(s - r)
            
            strategy_counts = np.array(strategy_counts)
            random_counts = np.array(random_counts)
            differences = np.array(differences)
            
            # Estatísticas
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            ci_lower = np.percentile(differences, 2.5)
            ci_upper = np.percentile(differences, 97.5)
            
            # Teste t pareado
            t_stat, p_value = stats.ttest_rel(strategy_counts, random_counts)
            
            # Teste Wilcoxon (não-paramétrico)
            try:
                w_stat, w_pvalue = stats.wilcoxon(strategy_counts, random_counts)
            except:
                w_pvalue = 1.0
            
            # Proporção de seeds com vantagem
            prop_positive = np.mean(differences > 0)
            
            # Interpretação
            if p_value < 0.01 and w_pvalue < 0.01:
                significance = "🔴 SIGNIFICATIVO"
            elif p_value < 0.05:
                significance = "🟡 MARGINAL"
            else:
                significance = "🟢 NÃO SIGNIFICATIVO"
            
            print(f"\n   {hits} PONTOS:")
            print(f"      Diferença média: {mean_diff:+.2f} ± {std_diff:.2f}")
            print(f"      IC 95%: [{ci_lower:+.2f}, {ci_upper:+.2f}]")
            print(f"      t-test p-value: {p_value:.4f}")
            print(f"      Wilcoxon p-value: {w_pvalue:.4f}")
            print(f"      Seeds com vantagem: {prop_positive:.1%}")
            print(f"      Conclusão: {significance}")
        
        # Análise de ROI
        strategy_rois = [r['strategy']['roi']['roi_pct'] for r in self.monte_carlo_results]
        random_rois = [r['random']['roi']['roi_pct'] for r in self.monte_carlo_results]
        
        print(f"\n   ROI FINANCEIRO:")
        print(f"      Estratégia: {np.mean(strategy_rois):.2f}% ± {np.std(strategy_rois):.2f}%")
        print(f"      Aleatório:  {np.mean(random_rois):.2f}% ± {np.std(random_rois):.2f}%")
        
        roi_diff = np.mean(strategy_rois) - np.mean(random_rois)
        print(f"      Diferença:  {roi_diff:+.2f}%")
        
        if roi_diff > 0:
            print(f"      ⚠️  Vantagem financeira MUITO pequena (se real)")
        else:
            print(f"      🟢 Sem vantagem financeira detectável")
    
    def run_walk_forward(self, train_size=200, test_size=50, step=50, jogos_por_concurso=10):
        """
        WALK-FORWARD VALIDATION
        
        Validação com janelas deslizantes:
        - Treina em janela de N concursos
        - Testa nos M seguintes
        - Desliza e repete
        
        Padrão profissional de validação temporal
        """
        print(f"\n{'='*70}")
        print(f"🚶 WALK-FORWARD VALIDATION")
        print(f"{'='*70}")
        print(f"📊 Configuração:")
        print(f"   Treino: {train_size} concursos")
        print(f"   Teste:  {test_size} concursos")
        print(f"   Passo:  {step} concursos")
        print(f"   Jogos:  {jogos_por_concurso} por concurso")
        
        self.walk_forward_results = []
        
        # Janelas deslizantes
        start = 0
        window = 0
        
        while start + train_size + test_size <= len(self.all_draws):
            window += 1
            
            train_data = self.all_draws[start:start + train_size]
            test_data = self.all_draws[start + train_size:start + train_size + test_size]
            
            # Acumuladores para esta janela
            strategy_hits = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
            random_hits = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
            
            for i, test_draw in enumerate(test_data):
                # Dados disponíveis: treino + testes anteriores nesta janela
                available_data = train_data + test_data[:i]
                
                # Gerar jogos
                strategy_games = self._generate_strategy_games(
                    available_data, jogos_por_concurso, seed=42 + window * 1000 + i
                )
                
                random_games = self._generate_random_constrained(
                    [d['dezenas'] for d in available_data],
                    jogos_por_concurso,
                    seed=42 + window * 1000 + i + 50000
                )
                
                # Verificar
                for game in strategy_games:
                    hits = self._count_hits(game, test_draw['dezenas'])
                    if hits >= 11:
                        strategy_hits[hits] += 1
                
                for game in random_games:
                    hits = self._count_hits(game, test_draw['dezenas'])
                    if hits >= 11:
                        random_hits[hits] += 1
            
            total_jogos = test_size * jogos_por_concurso
            
            self.walk_forward_results.append({
                'window': window,
                'train_start': train_data[0]['concurso'] if train_data else 0,
                'train_end': train_data[-1]['concurso'] if train_data else 0,
                'test_start': test_data[0]['concurso'] if test_data else 0,
                'test_end': test_data[-1]['concurso'] if test_data else 0,
                'strategy_hits': strategy_hits.copy(),
                'random_hits': random_hits.copy(),
                'strategy_total': sum(strategy_hits.values()),
                'random_total': sum(random_hits.values()),
                'strategy_roi': self._calculate_roi(strategy_hits, total_jogos),
                'random_roi': self._calculate_roi(random_hits, total_jogos)
            })
            
            start += step
        
        print(f"\n   ✅ {len(self.walk_forward_results)} janelas completadas")
        
        # Analisar
        self._analyze_walk_forward()
        
        return self.walk_forward_results
    
    def _analyze_walk_forward(self):
        """Analisa resultados do walk-forward"""
        if not self.walk_forward_results:
            return
        
        print(f"\n{'='*70}")
        print(f"📊 ANÁLISE WALK-FORWARD")
        print(f"{'='*70}")
        
        # Consolidar por faixa
        for hits in [11, 12, 13, 14, 15]:
            strategy_counts = [r['strategy_hits'].get(hits, 0) for r in self.walk_forward_results]
            random_counts = [r['random_hits'].get(hits, 0) for r in self.walk_forward_results]
            
            strategy_mean = np.mean(strategy_counts)
            random_mean = np.mean(random_counts)
            
            # Teste de Wilcoxon pareado
            try:
                w_stat, w_pvalue = stats.wilcoxon(strategy_counts, random_counts)
            except:
                w_pvalue = 1.0
            
            # Consistência (em quantas janelas a estratégia foi melhor)
            n_better = sum(1 for s, r in zip(strategy_counts, random_counts) if s > r)
            n_worse = sum(1 for s, r in zip(strategy_counts, random_counts) if s < r)
            n_equal = sum(1 for s, r in zip(strategy_counts, random_counts) if s == r)
            
            print(f"\n   {hits} PONTOS:")
            print(f"      Estratégia: {strategy_mean:.1f} ± {np.std(strategy_counts):.1f}")
            print(f"      Aleatório:  {random_mean:.1f} ± {np.std(random_counts):.1f}")
            print(f"      Janelas melhor: {n_better} | Pior: {n_worse} | Igual: {n_equal}")
            print(f"      Wilcoxon p-value: {w_pvalue:.4f}")
            
            if w_pvalue < 0.05:
                print(f"      🟡 DIFERENÇA SIGNIFICATIVA")
            else:
                print(f"      🟢 SEM DIFERENÇA SIGNIFICATIVA")
        
        # Consistência geral
        strategy_totals = [r['strategy_total'] for r in self.walk_forward_results]
        random_totals = [r['random_total'] for r in self.walk_forward_results]
        
        n_better_overall = sum(1 for s, r in zip(strategy_totals, random_totals) if s > r)
        
        print(f"\n   CONSISTÊNCIA GERAL:")
        print(f"      Estratégia melhor em {n_better_overall}/{len(self.walk_forward_results)} janelas")
        print(f"      ({n_better_overall/len(self.walk_forward_results)*100:.1f}%)")
        
        if n_better_overall > len(self.walk_forward_results) * 0.6:
            print(f"      🟡 Vantagem consistente (mas pode ser pequena)")
        else:
            print(f"      🟢 Sem vantagem consistente")
    
    def visualize_monte_carlo(self, output_dir='graficos_monte_carlo'):
        """Visualiza resultados do Monte Carlo"""
        if not self.monte_carlo_results:
            print("⚠️  Execute run_monte_carlo_backtest primeiro!")
            return
        
        print(f"\n🎨 GERANDO VISUALIZAÇÕES...")
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribuição da diferença (11 pontos)
        ax = axes[0, 0]
        diffs_11 = [r['strategy']['hits'].get(11, 0) - r['random']['hits'].get(11, 0) 
                    for r in self.monte_carlo_results]
        
        ax.hist(diffs_11, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=np.mean(diffs_11), color='red', linestyle='--', 
                  label=f'Média: {np.mean(diffs_11):+.1f}')
        ax.set_xlabel('Diferença (Estratégia - Aleatório)')
        ax.set_ylabel('Frequência')
        ax.set_title(f'Distribuição da Vantagem - 11 pontos\n({len(self.monte_carlo_results)} seeds)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distribuição da diferença (12 pontos)
        ax = axes[0, 1]
        diffs_12 = [r['strategy']['hits'].get(12, 0) - r['random']['hits'].get(12, 0) 
                    for r in self.monte_carlo_results]
        
        ax.hist(diffs_12, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(x=np.mean(diffs_12), color='red', linestyle='--', 
                  label=f'Média: {np.mean(diffs_12):+.1f}')
        ax.set_xlabel('Diferença')
        ax.set_ylabel('Frequência')
        ax.set_title(f'Distribuição da Vantagem - 12 pontos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Boxplot comparativo
        ax = axes[0, 2]
        data_11 = [
            [r['strategy']['hits'].get(11, 0) for r in self.monte_carlo_results],
            [r['random']['hits'].get(11, 0) for r in self.monte_carlo_results]
        ]
        
        bp = ax.boxplot(data_11, labels=['Estratégia', 'Aleatório'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('orange')
        ax.set_ylabel('Prêmios (11 pontos)')
        ax.set_title('Comparação de Distribuições - 11 pontos')
        ax.grid(True, alpha=0.3)
        
        # 4. Evolução do p-value com mais seeds
        ax = axes[1, 0]
        cumulative_pvalues = []
        for i in range(10, len(self.monte_carlo_results) + 1):
            subset_s = [r['strategy']['hits'].get(11, 0) for r in self.monte_carlo_results[:i]]
            subset_r = [r['random']['hits'].get(11, 0) for r in self.monte_carlo_results[:i]]
            try:
                _, p = stats.wilcoxon(subset_s, subset_r)
            except:
                p = 1.0
            cumulative_pvalues.append(p)
        
        ax.plot(range(10, len(self.monte_carlo_results) + 1), cumulative_pvalues, 
               color='purple', linewidth=2)
        ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        ax.set_xlabel('Número de Seeds')
        ax.set_ylabel('p-value (Wilcoxon)')
        ax.set_title('Estabilização do p-value com Mais Seeds')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. ROI comparativo
        ax = axes[1, 1]
        strategy_rois = [r['strategy']['roi']['roi_pct'] for r in self.monte_carlo_results]
        random_rois = [r['random']['roi']['roi_pct'] for r in self.monte_carlo_results]
        
        ax.hist(strategy_rois, bins=30, alpha=0.7, label='Estratégia', color='blue')
        ax.hist(random_rois, bins=30, alpha=0.7, label='Aleatório', color='orange')
        ax.axvline(x=np.mean(strategy_rois), color='blue', linestyle='--')
        ax.axvline(x=np.mean(random_rois), color='orange', linestyle='--')
        ax.set_xlabel('ROI (%)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição do ROI Financeiro')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Walk-forward (se disponível)
        ax = axes[1, 2]
        if self.walk_forward_results:
            windows = [r['window'] for r in self.walk_forward_results]
            strategy_totals = [r['strategy_total'] for r in self.walk_forward_results]
            random_totals = [r['random_total'] for r in self.walk_forward_results]
            
            ax.plot(windows, strategy_totals, 'bo-', label='Estratégia', linewidth=2)
            ax.plot(windows, random_totals, 'o-', color='orange', label='Aleatório', linewidth=2)
            ax.set_xlabel('Janela')
            ax.set_ylabel('Total de Prêmios')
            ax.set_title('Walk-Forward: Consistência Temporal')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Walk-forward não executado', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Walk-Forward (não disponível)')
        
        plt.suptitle('Monte Carlo Backtest - Validação Estatística Completa', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/monte_carlo_analysis.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Gráfico adicional: significância por faixa
        fig, ax = plt.subplots(figsize=(12, 6))
        
        faixas = [11, 12, 13, 14, 15]
        p_values = []
        
        for hits in faixas:
            s = [r['strategy']['hits'].get(hits, 0) for r in self.monte_carlo_results]
            r = [r['random']['hits'].get(hits, 0) for r in self.monte_carlo_results]
            try:
                _, p = stats.wilcoxon(s, r)
            except:
                p = 1.0
            p_values.append(p)
        
        colors = ['green' if p > 0.05 else 'yellow' if p > 0.01 else 'red' 
                 for p in p_values]
        ax.bar(faixas, p_values, color=colors, edgecolor='black', alpha=0.8)
        ax.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        ax.axhline(y=0.01, color='darkred', linestyle=':', label='p=0.01')
        ax.set_xlabel('Pontos')
        ax.set_ylabel('p-value')
        ax.set_title('Significância Estatística por Faixa de Premiação')
        ax.set_xticks(faixas)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/significancia_por_faixa.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
    
    def generate_final_report(self, output_dir='relatorio_final'):
        """Relatório final completo"""
        print(f"\n📄 GERANDO RELATÓRIO FINAL...")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'{output_dir}/backtest_final_{timestamp}.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RELATÓRIO FINAL DE BACKTEST - LOTOFÁCIL\n")
            f.write("Validação Científica Completa\n")
            f.write("="*70 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n")
            
            # Monte Carlo
            if self.monte_carlo_results:
                f.write("MONTE CARLO BACKTEST\n")
                f.write("-"*40 + "\n")
                f.write(f"Simulações: {len(self.monte_carlo_results)}\n\n")
                
                for hits in [11, 12, 13, 14, 15]:
                    s = [r['strategy']['hits'].get(hits, 0) for r in self.monte_carlo_results]
                    r = [r['random']['hits'].get(hits, 0) for r in self.monte_carlo_results]
                    
                    try:
                        _, p = stats.wilcoxon(s, r)
                    except:
                        p = 1.0
                    
                    f.write(f"{hits} pontos:\n")
                    f.write(f"  Estratégia: {np.mean(s):.1f} ± {np.std(s):.1f}\n")
                    f.write(f"  Aleatório:  {np.mean(r):.1f} ± {np.std(r):.1f}\n")
                    f.write(f"  p-value: {p:.4f}\n\n")
            
            # Walk-forward
            if self.walk_forward_results:
                f.write("\nWALK-FORWARD VALIDATION\n")
                f.write("-"*40 + "\n")
                f.write(f"Janelas: {len(self.walk_forward_results)}\n\n")
                
                n_better = sum(1 for r in self.walk_forward_results 
                             if r['strategy_total'] > r['random_total'])
                f.write(f"Estratégia melhor em {n_better}/{len(self.walk_forward_results)} janelas\n")
                f.write(f"Consistência: {n_better/len(self.walk_forward_results)*100:.1f}%\n\n")
            
            # Conclusão
            f.write("\n" + "="*70 + "\n")
            f.write("CONCLUSÃO FINAL\n")
            f.write("="*70 + "\n\n")
            
            # Determinar se houve vantagem significativa
            if self.monte_carlo_results:
                p_values = []
                for hits in [11, 12, 13, 14, 15]:
                    s = [r['strategy']['hits'].get(hits, 0) for r in self.monte_carlo_results]
                    r = [r['random']['hits'].get(hits, 0) for r in self.monte_carlo_results]
                    try:
                        _, p = stats.wilcoxon(s, r)
                    except:
                        p = 1.0
                    p_values.append(p)
                
                if all(p > 0.05 for p in p_values):
                    f.write("✅ NENHUMA VANTAGEM ESTATISTICAMENTE SIGNIFICATIVA\n\n")
                    f.write("A estratégia NÃO demonstrou superioridade\n")
                    f.write("sobre o baseline aleatório controlado.\n\n")
                    f.write("Isso é ESPERADO em loterias justas:\n")
                    f.write("- Todos os jogos têm a mesma probabilidade\n")
                    f.write("- Filtros estruturais não alteram a probabilidade fundamental\n")
                else:
                    f.write("⚠️  POSSÍVEL VANTAGEM DETECTADA\n\n")
                    f.write("Recomenda-se investigação adicional:\n")
                    f.write("- Aumentar número de simulações\n")
                    f.write("- Testar em períodos diferentes\n")
                    f.write("- Verificar possível overfitting\n")
            
            f.write("\n⚠️  DISCLAIMER:\n")
            f.write("Este backtest usa apenas dados passados.\n")
            f.write("Resultados passados não garantem resultados futuros.\n")
            f.write("Loteria é um jogo de azar. Jogue com responsabilidade.\n")
        
        print(f"✅ Relatório: {report_path}")
        return report_path


def main():
    """EXECUÇÃO PRINCIPAL"""
    print("="*70)
    print("🔬 BACKTEST AVANÇADO - LOTOFÁCIL v2.0")
    print("   Monte Carlo + Walk-Forward + Baseline Controlada")
    print("="*70)
    
    backtest = AdvancedBacktest()
    
    # Configuração
    n_concursos = 300
    jogos_por_concurso = 10
    n_seeds = 50  # Reduzido para teste rápido
    
    print(f"\n📋 CONFIGURAÇÃO:")
    print(f"   Concursos por backtest: {n_concursos}")
    print(f"   Jogos por concurso: {jogos_por_concurso}")
    print(f"   Seeds Monte Carlo: {n_seeds}")
    print(f"   Total simulações: {n_seeds * n_concursos * jogos_por_concurso:,}")
    
    # Perguntar se quer executar completo
    print(f"\n⏱️  Tempo estimado: {n_seeds * 2} segundos")
    resposta = input("▶️  Executar? (S/n): ").strip().lower()
    
    if resposta == 'n':
        print("❌ Cancelado.")
        return
    
    # 1. Monte Carlo
    backtest.run_monte_carlo_backtest(n_concursos, jogos_por_concurso, n_seeds)
    
    # 2. Walk-forward (mais leve)
    backtest.run_walk_forward(
        train_size=200,
        test_size=30,
        step=50,
        jogos_por_concurso=jogos_por_concurso
    )
    
    # 3. Visualizações
    backtest.visualize_monte_carlo()
    
    # 4. Relatório
    backtest.generate_final_report()
    
    print(f"\n{'='*70}")
    print(f"✅ BACKTEST CONCLUÍDO!")
    print(f"📁 Resultados em:")
    print(f"   • graficos_monte_carlo/ - Visualizações")
    print(f"   • relatorio_final/ - Relatório detalhado")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
