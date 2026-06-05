#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v59
RESPOSTA AO IMPULSO: VALORES BRUTOS (SEM MÉDIA MÓVEL)

CORREÇÃO METODOLÓGICA:
✅ Mede o valor bruto dos concursos futuros (t+1, t+2, ...)
   em vez da média móvel.
✅ Elimina a inércia artificial da média móvel.
✅ Detecta se, após um evento extremo, os sorteios individuais
   regridem à média ou permanecem independentes.
✅ Mantém walk‑forward honesto (sem vazamento temporal).
"""

import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict
import os, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
MOLDURA_SET = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
PRIMES_SET = {2, 3, 5, 7, 11, 13, 17, 19, 23}

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
# EXTRAÇÃO DE PADRÕES ESTRUTURAIS
# ============================================================
def extrair_series(contests):
    """Extrai todas as séries estruturais."""
    series = {
        'pares': [],
        'moldura': [],
        'primos': [],
        'soma': [],
        'consecutivos': [],
        'amplitude': []
    }
    for c in contests:
        d = c['dezenas']
        series['pares'].append(sum(1 for x in d if x % 2 == 0))
        series['moldura'].append(sum(1 for x in d if x in MOLDURA_SET))
        series['primos'].append(sum(1 for x in d if x in PRIMES_SET))
        series['soma'].append(sum(d))
        series['consecutivos'].append(sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1))
        series['amplitude'].append(max(d) - min(d))
    
    return {k: np.array(v, dtype=float) for k, v in series.items()}

# ============================================================
# RESPOSTA AO IMPULSO COM VALORES BRUTOS
# ============================================================
def impulso_response_raw(series_dict, janelas_media=[5, 10, 20, 50], 
                         thresholds=[1.0, 2.0],
                         horizontes=[1, 2, 3, 5, 10, 20, 50]):
    """
    Para cada padrão e cada janela de média móvel:
    1. Calcula média e desvio padrão históricos (até o concurso atual).
    2. Detecta eventos onde a média móvel se afasta da média histórica
       por mais de threshold desvios padrão.
    3. Para cada evento, registra o VALOR BRUTO dos concursos seguintes
       (não a média móvel).
    """
    resultados = {}
    
    for nome, serie in series_dict.items():
        n = len(serie)
        resultados[nome] = {}
        
        for janela in janelas_media:
            # Calcular média móvel
            media_movel = np.full(n, np.nan)
            for i in range(janela, n):
                media_movel[i] = np.mean(serie[i-janela:i])
            
            # Inicializar coletores
            eventos = defaultdict(list)  # chave: (threshold, direcao)
            
            for i in range(janela, n - max(horizontes)):
                if np.isnan(media_movel[i]):
                    continue
                
                # Média e desvio históricos ATÉ i (sem vazamento)
                hist_media = np.mean(serie[:i])
                hist_std = np.std(serie[:i])
                if hist_std == 0:
                    continue
                
                desvio = (media_movel[i] - hist_media) / hist_std
                
                # Verificar thresholds
                for thresh in thresholds:
                    if desvio > thresh:
                        # Coletar VALORES BRUTOS futuros
                        trajetoria = []
                        for h in horizontes:
                            if i + h < n:
                                trajetoria.append(serie[i + h])
                            else:
                                trajetoria.append(np.nan)
                        eventos[(thresh, 'alto')].append(trajetoria)
                    
                    elif desvio < -thresh:
                        trajetoria = []
                        for h in horizontes:
                            if i + h < n:
                                trajetoria.append(serie[i + h])
                            else:
                                trajetoria.append(np.nan)
                        eventos[(thresh, 'baixo')].append(trajetoria)
            
            # Agregar trajetórias
            resultados[nome][janela] = {}
            for (thresh, direcao), trajetorias in eventos.items():
                if len(trajetorias) == 0:
                    continue
                traj_array = np.array(trajetorias)
                media_traj = np.nanmean(traj_array, axis=0)
                std_traj = np.nanstd(traj_array, axis=0)
                n_eventos = len(trajetorias)
                resultados[nome][janela][(thresh, direcao)] = {
                    'media': media_traj,
                    'std': std_traj,
                    'n': n_eventos,
                    'horizontes': horizontes
                }
    
    return resultados

# ============================================================
# EXIBIÇÃO DOS RESULTADOS
# ============================================================
def exibir_resultados(resultados, series_dict):
    """Exibe as curvas de relaxação para cada padrão."""
    for nome in ['pares', 'moldura', 'primos']:
        serie = series_dict[nome]
        media_hist = np.mean(serie)
        std_hist = np.std(serie)
        
        print(f"\n{'='*70}")
        print(f"📊 PADRÃO: {nome.upper()} (média histórica={media_hist:.2f}, σ={std_hist:.2f})")
        print(f"{'='*70}")
        
        for janela in sorted(resultados[nome].keys()):
            print(f"\n   Janela de referência: {janela} concursos")
            for (thresh, direcao), dados in sorted(resultados[nome][janela].items()):
                if dados['n'] < 5:  # ignora amostras muito pequenas
                    continue
                print(f"\n   {'🔺' if direcao=='alto' else '🔻'} Desvio > {thresh}σ ({direcao}) — {dados['n']} eventos")
                print(f"   {'Horizonte':<10} {'Valor bruto':<15} {'Δ para média':<15}")
                print(f"   {'-'*40}")
                
                # Exibir todos os horizontes
                for i, h in enumerate(dados['horizontes']):
                    val = dados['media'][i]
                    if not np.isnan(val):
                        delta = val - media_hist
                        print(f"   t+{h:<8} {val:<15.4f} {delta:+.4f}")
                
                # Análise: em t+1, o valor já está próximo da média?
                if not np.isnan(dados['media'][0]):
                    delta_t1 = abs(dados['media'][0] - media_hist)
                    if delta_t1 < 0.5 * std_hist:
                        print(f"   ✅ Em t+1, o valor já está a menos de 0.5σ da média histórica.")
                        print(f"      Isso sugere independência: cada concurso é aproximadamente i.i.d.")
                    elif delta_t1 < 1.0 * std_hist:
                        print(f"   📊 Em t+1, o valor está entre 0.5σ e 1σ da média.")
                    else:
                        print(f"   🔍 Em t+1, o desvio ainda é >1σ. Possível persistência de curto prazo.")

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v59")
    print("   RESPOSTA AO IMPULSO: VALORES BRUTOS (SEM MÉDIA MÓVEL)")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    
    print(f"\n📂 {len(contests)} concursos")
    
    # Extrair séries
    series_dict = extrair_series(contests)
    print("📊 Séries extraídas: pares, moldura, primos, soma, consecutivos, amplitude")
    
    # Parâmetros
    janelas = [5, 10, 20, 50]
    thresholds = [1.0, 2.0]
    horizontes = [1, 2, 3, 5, 10, 20, 50]
    
    print(f"\n⚙️ Parâmetros:")
    print(f"   Janelas de referência: {janelas}")
    print(f"   Thresholds (σ): {thresholds}")
    print(f"   Horizontes: {horizontes}")
    
    # Executar análise
    print(f"\n🔄 Processando...")
    resultados = impulso_response_raw(series_dict, janelas, thresholds, horizontes)
    
    # Exibir resultados
    exibir_resultados(resultados, series_dict)
    
    # Conclusão interpretativa
    print(f"\n{'='*70}")
    print("📊 INTERPRETAÇÃO")
    print("="*70)
    print("Este teste mede o VALOR BRUTO de cada concurso futuro após")
    print("um evento extremo na média móvel. Não há inércia artificial.")
    print("Se em t+1 o valor já está próximo da média histórica, isso")
    print("indica que os sorteios são essencialmente independentes (i.i.d.).")
    print("Se o desvio persiste por vários concursos, pode haver memória")
    print("de curto prazo.")
    
    print("\n✅ Análise concluída.")

if __name__ == "__main__":
    main()
