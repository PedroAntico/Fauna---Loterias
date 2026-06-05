#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v57
TESTES DE REGRESSÃO À MÉDIA NOS PARES

TRÊS TESTES INDEPENDENTES:
✅ Teste A (Binário): se média recente > 7.2, prevê ≤7 pares;
   se < 7.2, prevê ≥8. Mede acurácia e p‑valor.
✅ Teste B (Correlação): correlação entre desvio atual e desvio futuro.
   Se negativa e significativa → regressão à média.
✅ Teste C (Extremos): quando a janela recente está > 8.0 ou < 6.5 pares,
   mede a média do próximo concurso.
✅ Walk‑forward honesto (sem vazamento temporal)
"""

import numpy as np
from scipy.stats import binomtest, pearsonr
import os, warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
MEDIA_TEORICA_PARES = 15 * 12 / 25   # 7.2

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
# EXTRAÇÃO DA SÉRIE DE PARES
# ============================================================
def extrair_serie_pares(contests):
    """Retorna um array numpy com o número de pares em cada concurso."""
    return np.array([sum(1 for x in c['dezenas'] if x % 2 == 0) for c in contests], dtype=float)

# ============================================================
# TESTE A: PREVISÃO BINÁRIA
# ============================================================
def teste_binario(serie, janela=20):
    """
    Para cada ponto t (janela .. n-1):
    - calcula a média dos últimos 'janela' valores (até t-1)
    - se média > 7.2, prevê que o próximo valor será ≤ 7
    - se média < 7.2, prevê que o próximo valor será ≥ 8
    - compara com o valor real em t
    """
    n = len(serie)
    acertos = 0
    total = 0
    
    for t in range(janela, n):
        media_recente = np.mean(serie[t-janela:t])
        valor_real = serie[t]
        
        if media_recente > MEDIA_TEORICA_PARES:
            previsao = 'baixo'  # espera ≤7
            acertou = (valor_real <= 7)
        elif media_recente < MEDIA_TEORICA_PARES:
            previsao = 'alto'   # espera ≥8
            acertou = (valor_real >= 8)
        else:
            continue  # exatamente na média, ignora
        
        total += 1
        if acertou:
            acertos += 1
    
    if total == 0:
        return 0.5, 0, 0, 1.0
    
    acuracia = acertos / total
    p_value = binomtest(acertos, total, 0.5, alternative='greater').pvalue
    
    return acuracia, acertos, total, p_value

# ============================================================
# TESTE B: CORRELAÇÃO DE DESVIOS
# ============================================================
def teste_correlacao(serie, janela=20):
    """
    Para cada t (janela .. n-1):
    - desvio_atual = média_recente - 7.2
    - desvio_futuro = valor_real - 7.2
    Mede a correlação entre esses dois vetores.
    """
    n = len(serie)
    desvios_atuais = []
    desvios_futuros = []
    
    for t in range(janela, n):
        media_recente = np.mean(serie[t-janela:t])
        valor_real = serie[t]
        desvios_atuais.append(media_recente - MEDIA_TEORICA_PARES)
        desvios_futuros.append(valor_real - MEDIA_TEORICA_PARES)
    
    if len(desvios_atuais) < 10:
        return 0.0, 1.0, 0
    
    corr, p_value = pearsonr(desvios_atuais, desvios_futuros)
    return corr, p_value, len(desvios_atuais)

# ============================================================
# TESTE C: ANÁLISE DE EXTREMOS
# ============================================================
def teste_extremos(serie, janela=20, limiar_alto=8.0, limiar_baixo=6.5):
    """
    Quando a média recente está acima de limiar_alto,
    verifica a média do próximo valor.
    Quando está abaixo de limiar_baixo, idem.
    """
    n = len(serie)
    altos = []
    baixos = []
    
    for t in range(janela, n):
        media_recente = np.mean(serie[t-janela:t])
        valor_real = serie[t]
        
        if media_recente > limiar_alto:
            altos.append(valor_real)
        elif media_recente < limiar_baixo:
            baixos.append(valor_real)
    
    resultado = {}
    if altos:
        resultado['media_pos_alto'] = np.mean(altos)
        resultado['n_alto'] = len(altos)
    else:
        resultado['media_pos_alto'] = None
        resultado['n_alto'] = 0
    
    if baixos:
        resultado['media_pos_baixo'] = np.mean(baixos)
        resultado['n_baixo'] = len(baixos)
    else:
        resultado['media_pos_baixo'] = None
        resultado['n_baixo'] = 0
    
    return resultado

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v57")
    print("   REGRESSÃO À MÉDIA NOS PARES – TRÊS TESTES")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    
    print(f"\n📂 {len(contests)} concursos")
    
    # Extrair série de pares
    serie_pares = extrair_serie_pares(contests)
    print(f"📊 Série de pares: {len(serie_pares)} valores")
    print(f"   Média histórica: {np.mean(serie_pares):.4f}")
    print(f"   Média teórica:   {MEDIA_TEORICA_PARES:.4f}")
    
    # Parâmetros
    janela = 20
    print(f"\n⚙️ Janela para média recente: {janela} concursos")
    
    # Teste A
    print("\n" + "="*70)
    print("TESTE A: PREVISÃO BINÁRIA")
    print("="*70)
    acuracia, acertos, total, p_val = teste_binario(serie_pares, janela)
    print(f"   Previsões: {total}")
    print(f"   Acertos: {acertos}")
    print(f"   Acurácia: {acuracia*100:.2f}%")
    print(f"   p‑valor (binomial): {p_val:.4f}")
    if p_val < 0.05:
        print(f"   🔍 Estatisticamente significativo (p < 0.05)")
    else:
        print(f"   📊 Não significativo")
    
    # Teste B
    print("\n" + "="*70)
    print("TESTE B: CORRELAÇÃO DE DESVIOS")
    print("="*70)
    corr, p_corr, n_pares = teste_correlacao(serie_pares, janela)
    print(f"   Pares (desvio atual, desvio futuro): {n_pares}")
    print(f"   Correlação de Pearson: {corr:.4f}")
    print(f"   p‑valor: {p_corr:.4f}")
    if corr < 0 and p_corr < 0.05:
        print(f"   🔍 Correlação negativa significativa → regressão à média detectada!")
    elif corr < 0:
        print(f"   📊 Correlação negativa, mas não significativa.")
    elif corr > 0:
        print(f"   📊 Correlação positiva (não indica regressão à média).")
    else:
        print(f"   📊 Correlação nula.")
    
    # Teste C
    print("\n" + "="*70)
    print("TESTE C: ANÁLISE DE EXTREMOS")
    print("="*70)
    extremos = teste_extremos(serie_pares, janela)
    print(f"   Após janela com média > 8.0 pares:")
    if extremos['media_pos_alto'] is not None:
        print(f"      Média do próximo concurso: {extremos['media_pos_alto']:.4f}")
        print(f"      Ocorrências: {extremos['n_alto']}")
    else:
        print(f"      Nenhuma ocorrência.")
    print(f"   Após janela com média < 6.5 pares:")
    if extremos['media_pos_baixo'] is not None:
        print(f"      Média do próximo concurso: {extremos['media_pos_baixo']:.4f}")
        print(f"      Ocorrências: {extremos['n_baixo']}")
    else:
        print(f"      Nenhuma ocorrência.")
    
    # Conclusão
    print("\n" + "="*70)
    print("📊 CONCLUSÃO")
    print("="*70)
    if p_val < 0.05 and corr < 0 and p_corr < 0.05:
        print("✅ Os três testes indicam regressão à média significativa nos pares.")
        print("   Este é o primeiro sinal robusto encontrado pelo laboratório.")
    elif p_val < 0.05 or (corr < 0 and p_corr < 0.05):
        print("📊 Alguns testes indicam regressão à média, mas não todos.")
        print("   São necessários mais experimentos para confirmar.")
    else:
        print("❌ Nenhum teste detectou regressão à média significativa.")
        print("   Os pares não mostram memória temporal explorável.")
    
    print("\n✅ Experimentos concluídos.")

if __name__ == "__main__":
    main()
