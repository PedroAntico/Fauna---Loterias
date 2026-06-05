#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v54
TESTE DE PADRÕES ESTRUTURAIS FORA DA AMOSTRA

OBJETIVO:
✅ Abandonar busca por trincas fixas
✅ Avaliar poder preditivo de padrões estáveis:
   - Pares, moldura, primos, soma, consecutivos, amplitude
   - Frequência individual das dezenas
   - Atraso (delay) das dezenas
✅ Walk‑forward com treino/teste independentes
✅ Comparação contra baseline aleatório
✅ Ranking de padrões com verdadeiro poder preditivo
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
PRIMES_SET = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA_SET = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}

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
# EXTRATOR DE PADRÕES ESTRUTURAIS
# ============================================================
def extract_structural_features(dezenas):
    """Extrai características estruturais de um jogo."""
    d = sorted(dezenas)
    return {
        'pares': sum(1 for x in d if x % 2 == 0),
        'moldura': sum(1 for x in d if x in MOLDURA_SET),
        'primos': sum(1 for x in d if x in PRIMES_SET),
        'soma': sum(d),
        'consecutivos': sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1),
        'amplitude': max(d) - min(d),
        'dezenas_presentes': set(d)
    }

def build_structural_series(contests):
    """Constrói séries temporais para cada padrão estrutural."""
    series = {
        'pares': [],
        'moldura': [],
        'primos': [],
        'soma': [],
        'consecutivos': [],
        'amplitude': []
    }
    
    # Frequência por dezena (acumulada)
    freq_acumulada = {d: [] for d in range(1, 26)}
    atraso = {d: [] for d in range(1, 26)}
    last_seen = {d: -1 for d in range(1, 26)}
    
    for idx, c in enumerate(contests):
        feat = extract_structural_features(c['dezenas'])
        for key in series:
            series[key].append(feat[key])
        
        present = feat['dezenas_presentes']
        for d in range(1, 26):
            # Frequência acumulada até aqui
            freq = sum(1 for past in contests[:idx+1] if d in past['dezenas'])
            freq_acumulada[d].append(freq / (idx+1))
            
            # Atraso
            if d in present:
                last_seen[d] = idx
            atraso[d].append(idx - last_seen[d])
    
    return series, freq_acumulada, atraso

# ============================================================
# AVALIADOR DE PADRÃO COM WALK‑FORWARD
# ============================================================
def evaluate_pattern_walk_forward(contests, pattern_name, pattern_series, 
                                  train_size=500, test_size=50, step=50):
    """
    Walk‑forward: tenta prever a direção do padrão no próximo bloco
    usando apenas informações do bloco de treino.
    """
    n = len(pattern_series)
    predictions = []
    actuals = []
    
    start = train_size
    while start + test_size <= n:
        train_data = pattern_series[start-train_size:start]
        test_data = pattern_series[start:start+test_size]
        
        # Característica do treino
        mean_train = np.mean(train_data)
        mean_long = np.mean(pattern_series[:start])
        
        # Previsão: reversão à média
        if mean_train > mean_long:
            pred_direction = -1  # prevê queda
        else:
            pred_direction = 1   # prevê alta
        
        # Realidade no teste
        mean_test = np.mean(test_data)
        actual_direction = 1 if mean_test > mean_train else -1
        
        predictions.append(pred_direction)
        actuals.append(actual_direction)
        
        start += step
    
    if len(predictions) == 0:
        return {'accuracy': 50.0, 'n_predictions': 0, 'p_value': 1.0}
    
    accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(predictions) * 100
    n_correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    
    # Significância contra 50% (acaso)
    p_value = binomtest(n_correct, len(predictions), 0.5, alternative='greater').pvalue
    
    return {
        'accuracy': accuracy,
        'n_predictions': len(predictions),
        'n_correct': n_correct,
        'p_value': p_value,
        'predictions': predictions,
        'actuals': actuals
    }

# ============================================================
# BASELINE ALEATÓRIO PARA COMPARAÇÃO
# ============================================================
def generate_random_series(n, pattern_type='binary'):
    """Gera uma série aleatória do mesmo tamanho."""
    if pattern_type == 'binary':
        return np.random.choice([0, 1], size=n)
    else:
        # Para séries contínuas, gera ruído branco
        return np.random.randn(n)

def evaluate_random_baseline(contests, pattern_name, n_simulations=100,
                             train_size=500, test_size=50, step=50):
    """Avalia o baseline aleatório para um padrão."""
    n = len(contests)
    accuracies = []
    
    for _ in range(n_simulations):
        # Gera série aleatória do mesmo tamanho
        random_series = np.random.randn(n)
        result = evaluate_pattern_walk_forward(contests, pattern_name, random_series,
                                               train_size, test_size, step)
        accuracies.append(result['accuracy'])
    
    return np.mean(accuracies), np.std(accuracies)

# ============================================================
# TESTE PRINCIPAL DE PADRÕES ESTRUTURAIS
# ============================================================
def test_structural_patterns(contests, train_size=500, test_size=50, step=50, n_baseline=200):
    """
    Testa todos os padrões estruturais e ranqueia por poder preditivo real.
    """
    print("\n" + "="*70)
    print("🔬 TESTE DE PADRÕES ESTRUTURAIS FORA DA AMOSTRA")
    print("="*70)
    print(f"   Walk‑forward: treino {train_size}, teste {test_size}, passo {step}")
    print(f"   Baseline: {n_baseline} simulações aleatórias por padrão\n")
    
    # Extrair séries estruturais
    series, freq_dezenas, atraso_dezenas = build_structural_series(contests)
    
    # Padrões a testar
    patterns_to_test = {
        'Pares': ('estrutural', series['pares']),
        'Moldura': ('estrutural', series['moldura']),
        'Primos': ('estrutural', series['primos']),
        'Soma': ('estrutural', series['soma']),
        'Consecutivos': ('estrutural', series['consecutivos']),
        'Amplitude': ('estrutural', series['amplitude']),
    }
    
    # Adicionar frequência das dezenas mais estáveis
    for d in [20, 12, 4, 11, 10, 25, 18, 21]:  # baseado nos achados anteriores
        patterns_to_test[f'Freq Dezena {d}'] = ('frequencia', freq_dezenas[d])
    
    # Adicionar atraso das mesmas dezenas
    for d in [20, 12, 4, 11, 10, 25]:
        patterns_to_test[f'Atraso Dezena {d}'] = ('atraso', atraso_dezenas[d])
    
    results = {}
    baseline_results = {}
    
    # Avaliar cada padrão
    for name, (tipo, serie) in tqdm(patterns_to_test.items(), desc="Avaliando padrões"):
        result = evaluate_pattern_walk_forward(contests, name, serie,
                                               train_size, test_size, step)
        results[name] = result
        
        # Baseline aleatório específico para este padrão
        mean_base, std_base = evaluate_random_baseline(contests, name, n_baseline,
                                                       train_size, test_size, step)
        baseline_results[name] = (mean_base, std_base)
    
    # Calcular z‑score e ranquear
    final_ranking = []
    for name, res in results.items():
        mean_base, std_base = baseline_results[name]
        z = (res['accuracy'] - mean_base) / std_base if std_base > 0 else 0.0
        pct = 0.0
        if std_base > 0:
            # Estima percentil assumindo distribuição normal
            from scipy.stats import norm
            pct = norm.cdf(z) * 100
        
        final_ranking.append({
            'pattern': name,
            'accuracy': res['accuracy'],
            'baseline_mean': mean_base,
            'z_score': z,
            'percentile': pct,
            'n_predictions': res['n_predictions'],
            'p_value': res['p_value']
        })
    
    # Ordenar por z‑score
    final_ranking.sort(key=lambda x: x['z_score'], reverse=True)
    
    # Exibir resultados
    print(f"\n📊 RANKING DE PODER PREDITIVO REAL:")
    print(f"{'Padrão':<20} {'Acurácia':<10} {'Baseline':<10} {'Z‑score':<10} {'%ile':<8} {'p‑value':<10} {'Avaliação'}")
    print("-" * 85)
    
    for item in final_ranking:
        if item['z_score'] > 2.0:
            avaliacao = "🔍 FORTE"
        elif item['z_score'] > 1.0:
            avaliacao = "📊 Leve"
        elif item['z_score'] > 0:
            avaliacao = "📊 Marginal"
        else:
            avaliacao = "❌ Nulo"
        
        print(f"{item['pattern']:<20} {item['accuracy']:<10.1f}% {item['baseline_mean']:<10.1f}% "
              f"{item['z_score']:<10.2f} {item['percentile']:<8.1f}% {item['p_value']:<10.4f} {avaliacao}")
    
    # Resumo
    strong = [item for item in final_ranking if item['z_score'] > 2.0]
    any_strong = len(strong) > 0
    
    print(f"\n📊 RESUMO:")
    print(f"   Padrões com z > 2.0: {len(strong)}")
    if any_strong:
        print(f"   Melhores: {[s['pattern'] for s in strong]}")
    else:
        print(f"   Nenhum padrão mostrou poder preditivo forte fora da amostra.")
        print(f"   Isso sugere que a Lotofácil se comporta de forma essencialmente aleatória")
        print(f"   para as características estruturais testadas.")
    
    return final_ranking

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v54")
    print("   PODER PREDITIVO DE PADRÕES ESTRUTURAIS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")

    # Parâmetros do walk‑forward
    print("\nParâmetros do walk‑forward:")
    try:
        train_size = int(input("   Tamanho do treino [500]: ").strip() or "500")
        test_size = int(input("   Tamanho do teste [50]: ").strip() or "50")
        step = int(input("   Passo [50]: ").strip() or "50")
    except:
        train_size, test_size, step = 500, 50, 50

    # Executar teste
    results = test_structural_patterns(contests, train_size, test_size, step)

    # Perguntar se quer ver detalhes de algum padrão
    ver_detalhes = input("\nVer detalhes de algum padrão? (nome ou ENTER para sair): ").strip()
    if ver_detalhes:
        # Reconstruir séries para mostrar
        series, _, _ = build_structural_series(contests)
        # Mostrar os últimos valores
        if ver_detalhes == 'Pares':
            print(f"   Últimos 10 valores de Pares: {series['pares'][-10:]}")
        elif ver_detalhes == 'Moldura':
            print(f"   Últimos 10 valores de Moldura: {series['moldura'][-10:]}")
        elif ver_detalhes == 'Primos':
            print(f"   Últimos 10 valores de Primos: {series['primos'][-10:]}")
        elif ver_detalhes == 'Soma':
            print(f"   Últimos 10 valores de Soma: {series['soma'][-10:]}")
        elif ver_detalhes == 'Consecutivos':
            print(f"   Últimos 10 valores de Consecutivos: {series['consecutivos'][-10:]}")
        elif ver_detalhes == 'Amplitude':
            print(f"   Últimos 10 valores de Amplitude: {series['amplitude'][-10:]}")
        else:
            print(f"   Padrão '{ver_detalhes}' não reconhecido.")

    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
