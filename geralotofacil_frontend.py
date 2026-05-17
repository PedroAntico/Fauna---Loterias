#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRONTEND HÍBRIDO - PREFERÊNCIAS HUMANAS + OTIMIZAÇÃO ESTRUTURAL
================================================================
Versão 2.0 - Filtros Flexíveis com Scoring Contínuo

PRINCÍPIO:
✅ NÃO gera jogos com regras (destrói otimização)
✅ FILTRA e RANKEIA jogos já otimizados da fronteira
✅ Constraints viram SCORES (não eliminações)
✅ Preserva geometria do espaço combinatório
✅ Sistema híbrido: estrutural (70%) + humano (30%)
"""

import numpy as np
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from math import comb

# ============================================================
# CONJUNTOS MATEMÁTICOS DA LOTOFÁCIL
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}

MOLDURA = {
    1, 2, 3, 4, 5,      # linha 1 completa
    6, 10,               # linha 2 bordas
    11, 15,              # linha 3 bordas
    16, 20,              # linha 4 bordas
    21, 22, 23, 24, 25   # linha 5 completa
}

CENTRO = {7, 8, 9, 12, 13, 14, 17, 18, 19}

QUADRANTES = {
    'Q1': {1, 2, 3, 4, 5},
    'Q2': {6, 7, 8, 9, 10},
    'Q3': {11, 12, 13, 14, 15},
    'Q4': {16, 17, 18, 19, 20},
    'Q5': {21, 22, 23, 24, 25}
}

# Distribuições típicas históricas (referência)
TYPICAL_RANGES = {
    'pares': (6, 9),
    'impares': (6, 9),
    'primos': (3, 6),
    'moldura': (7, 10),
    'centro': (5, 8),
    'soma': (170, 220),
    'repetidas': (7, 10),
    'consecutivos': (3, 7)
}


# ============================================================
# CARREGAMENTO DA FRONTEIRA
# ============================================================

def load_pareto_frontier(filename='pareto_frontier.json'):
    """Carrega fronteira de Pareto exportada pelo motor"""
    if not os.path.exists(filename):
        print(f"❌ Arquivo {filename} não encontrado!")
        print(f"   Execute primeiro: python system_facil.py")
        return None
    
    print(f"📂 Carregando {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   ✅ {data['metadata']['n_solutions']} soluções | "
          f"{data['metadata']['n_games_per_pool']} jogos/pool")
    
    return data


# ============================================================
# CLASSIFICAÇÃO DE PERFIS ESTRUTURAIS
# ============================================================

def classify_profiles(signals_list):
    """Classifica pools em perfis com normalização populacional"""
    if not signals_list:
        return []
    
    keys = signals_list[0].keys()
    ranges = {}
    for key in keys:
        values = [s[key] for s in signals_list]
        ranges[key] = (np.min(values), np.max(values))
    
    profiles = []
    for signals in signals_list:
        norm = {}
        for key in keys:
            vmin, vmax = ranges[key]
            norm[key] = (signals[key] - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        
        scores = {
            'conservador': (
                (1 - norm.get('pos_entropy', 0.5)) * 1.5 +
                norm.get('compressibility', 0.5) * 1.5 +
                (1 - norm.get('johnson_avg', 0.5)) * 1.0
            ),
            'caotico': (
                norm.get('johnson_min', 0.5) * 1.5 +
                norm.get('johnson_avg', 0.5) * 1.5 +
                norm.get('pos_entropy', 0.5) * 1.0
            ),
            'cobertura': (
                norm.get('pair_coverage', 0.5) * 1.5 +
                (1 - norm.get('covering_radius', 0.5)) * 1.5 +
                norm.get('sphere_packing', 0.5) * 1.0
            ),
            'balanceado': (
                norm.get('pair_coverage', 0.5) * 1.0 +
                norm.get('johnson_min', 0.5) * 1.0 +
                (1 - norm.get('covering_radius', 0.5)) * 1.0 +
                norm.get('pos_entropy', 0.5) * 1.0
            )
        }
        
        profiles.append({
            'profile': max(scores, key=scores.get),
            'scores': scores,
            'signals': signals
        })
    
    return profiles


# ============================================================
# SCORING DE JOGOS (ESTRUTURAL)
# ============================================================

def structural_game_score(game, pool, signals):
    """
    Score estrutural de um jogo dentro do pool
    
    Considera:
    - Diversidade (distância para outros jogos)
    - Balanceamento par/ímpar
    - Distribuição alto/baixo
    - Representatividade do perfil
    """
    score = 0.0
    
    # 1. Distância média para outros jogos (DIVERSIDADE)
    distances = []
    for other in pool:
        if game != other:
            common = len(set(game) & set(other))
            distances.append(15 - common)
    
    if distances:
        avg_dist = np.mean(distances)
        score += (avg_dist / 10) * 0.35  # Normalizado
    
    # 2. Balanceamento par/ímpar (7-8 é ideal)
    pares = sum(1 for d in game if d % 2 == 0)
    balance = 1.0 - abs(pares - 7.5) / 7.5
    score += balance * 0.25
    
    # 3. Distribuição alto/baixo
    baixas = sum(1 for d in game if d <= 12)
    spread = 1.0 - abs(baixas - 7.5) / 7.5
    score += spread * 0.20
    
    # 4. Entropia local (variedade de dezenas)
    unique_quadrants = len(set((d-1)//5 for d in game))
    score += (unique_quadrants / 5) * 0.20
    
    return score


# ============================================================
# SCORING DE PREFERÊNCIAS HUMANAS (FLEXÍVEL)
# ============================================================

def human_constraint_score(game, preferences, ultimo_concurso=None):
    """
    Score contínuo baseado em preferências humanas
    
    NÃO elimina jogos - apenas ajusta o score
    Quanto mais próximo das preferências, maior o score
    
    Args:
        game: Lista de 15 dezenas
        preferences: Dict com preferências do usuário
        ultimo_concurso: Lista do último concurso (para repetidas)
    
    Returns:
        float: Score 0-1 (1 = perfeito match)
    """
    scores = []
    weights = []
    
    # 1. Pares (com tolerância)
    if 'pares' in preferences:
        target = preferences['pares']
        tolerance = preferences.get('pares_tolerance', 1)
        actual = sum(1 for d in game if d % 2 == 0)
        dist = abs(actual - target)
        
        if dist <= tolerance:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tolerance) / 7))
        weights.append(1.5)
    
    # 2. Primos (com tolerância)
    if 'primos' in preferences:
        target = preferences['primos']
        tolerance = preferences.get('primos_tolerance', 1)
        actual = sum(1 for d in game if d in PRIMES)
        dist = abs(actual - target)
        
        if dist <= tolerance:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tolerance) / 5))
        weights.append(1.2)
    
    # 3. Moldura (com tolerância)
    if 'moldura' in preferences:
        target = preferences['moldura']
        tolerance = preferences.get('moldura_tolerance', 1)
        actual = sum(1 for d in game if d in MOLDURA)
        dist = abs(actual - target)
        
        if dist <= tolerance:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tolerance) / 7))
        weights.append(1.0)
    
    # 4. Centro (com tolerância)
    if 'centro' in preferences:
        target = preferences['centro']
        tolerance = preferences.get('centro_tolerance', 1)
        actual = sum(1 for d in game if d in CENTRO)
        dist = abs(actual - target)
        
        if dist <= tolerance:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tolerance) / 5))
        weights.append(0.8)
    
    # 5. Repetidas do último concurso
    if 'repetidas' in preferences and ultimo_concurso:
        target = preferences['repetidas']
        tolerance = preferences.get('repetidas_tolerance', 1)
        actual = len(set(game) & set(ultimo_concurso))
        dist = abs(actual - target)
        
        if dist <= tolerance:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tolerance) / 7))
        weights.append(1.3)
    
    # 6. Soma (com faixa)
    if 'soma_min' in preferences or 'soma_max' in preferences:
        soma = sum(game)
        soma_min = preferences.get('soma_min', 170)
        soma_max = preferences.get('soma_max', 220)
        
        if soma_min <= soma <= soma_max:
            scores.append(1.0)
        else:
            dist = min(abs(soma - soma_min), abs(soma - soma_max))
            scores.append(max(0, 1.0 - dist / 50))
        weights.append(0.7)
    
    # 7. Consecutivos (com máximo)
    if 'max_consecutivos' in preferences:
        max_cons = preferences['max_consecutivos']
        d = sorted(game)
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
        
        if cons <= max_cons:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (cons - max_cons) / 5))
        weights.append(0.9)
    
    # 8. Dezenas fixas (obrigatório - score 0 se não tiver)
    if 'fixas' in preferences and preferences['fixas']:
        fixas = set(preferences['fixas'])
        if not fixas.issubset(set(game)):
            return 0.0  # Jogo não atende requisito obrigatório
        # Bônus por ter as fixas
        scores.append(1.0)
        weights.append(2.0)
    
    # 9. Dezenas excluídas (obrigatório - score 0 se tiver)
    if 'excluidas' in preferences and preferences['excluidas']:
        excluidas = set(preferences['excluidas'])
        if excluidas & set(game):
            return 0.0  # Jogo contém dezena proibida
    
    # Média ponderada
    if not scores:
        return 1.0  # Sem preferências = score máximo
    
    total_weight = sum(weights)
    weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    return weighted_score


# ============================================================
# RANKEAMENTO HÍBRIDO
# ============================================================

def hybrid_ranking(pool, signals, preferences, ultimo_concurso=None, 
                   structural_weight=0.7, human_weight=0.3):
    """
    Ranking HÍBRIDO: estrutural + preferências humanas
    
    Args:
        pool: Lista de jogos
        signals: Sinais estruturais do pool
        preferences: Preferências do usuário
        ultimo_concurso: Último concurso (para repetidas)
        structural_weight: Peso do score estrutural
        human_weight: Peso das preferências humanas
    
    Returns:
        list: (score_final, game, structural_score, human_score)
    """
    ranked = []
    
    for game in pool:
        # Score estrutural
        struct_score = structural_game_score(game, pool, signals)
        
        # Score humano
        human_score = human_constraint_score(game, preferences, ultimo_concurso)
        
        # Score final híbrido
        final_score = struct_score * structural_weight + human_score * human_weight
        
        ranked.append((final_score, sorted(game), struct_score, human_score))
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ============================================================
# COLETA DE PREFERÊNCIAS
# ============================================================

def collect_user_preferences(ultimo_concurso=None):
    """
    Interface interativa para coletar preferências
    
    Todas as preferências são OPCIONAIS (ENTER = sem restrição)
    """
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
    print(f"{'='*60}")
    print(f"\n💡 Pressione ENTER para ignorar qualquer preferência")
    print(f"   (O sistema usará apenas otimização estrutural)")
    
    prefs = {}
    
    # 1. Pares
    print(f"\n📊 PARES (típico: 6-9)")
    print(f"   Quantos números PARES você prefere?")
    pares_input = input(f"   Pares [ENTER=pular]: ").strip()
    if pares_input:
        try:
            prefs['pares'] = int(pares_input)
            tol = input(f"   Tolerância (±) [1]: ").strip()
            prefs['pares_tolerance'] = int(tol) if tol else 1
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # 2. Primos
    print(f"\n🔢 PRIMOS (típico: 3-6)")
    print(f"   Primos na Lotofácil: {sorted(PRIMES)}")
    primos_input = input(f"   Quantos PRIMOS? [ENTER=pular]: ").strip()
    if primos_input:
        try:
            prefs['primos'] = int(primos_input)
            tol = input(f"   Tolerância (±) [1]: ").strip()
            prefs['primos_tolerance'] = int(tol) if tol else 1
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # 3. Moldura
    print(f"\n🖼️  MOLDURA (típico: 7-10)")
    print(f"   Moldura: {sorted(MOLDURA)}")
    moldura_input = input(f"   Quantos na MOLDURA? [ENTER=pular]: ").strip()
    if moldura_input:
        try:
            prefs['moldura'] = int(moldura_input)
            tol = input(f"   Tolerância (±) [1]: ").strip()
            prefs['moldura_tolerance'] = int(tol) if tol else 1
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # 4. Centro
    print(f"\n🎯 CENTRO (típico: 5-8)")
    print(f"   Centro: {sorted(CENTRO)}")
    centro_input = input(f"   Quantos no CENTRO? [ENTER=pular]: ").strip()
    if centro_input:
        try:
            prefs['centro'] = int(centro_input)
            tol = input(f"   Tolerância (±) [1]: ").strip()
            prefs['centro_tolerance'] = int(tol) if tol else 1
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # 5. Repetidas
    if ultimo_concurso:
        print(f"\n🔄 REPETIDAS DO ÚLTIMO CONCURSO")
        print(f"   Último concurso: {sorted(ultimo_concurso)}")
        print(f"   (típico: 7-10 repetidas)")
        rep_input = input(f"   Quantas REPETIDAS? [ENTER=pular]: ").strip()
        if rep_input:
            try:
                prefs['repetidas'] = int(rep_input)
                tol = input(f"   Tolerância (±) [1]: ").strip()
                prefs['repetidas_tolerance'] = int(tol) if tol else 1
            except ValueError:
                print(f"   ⚠️  Valor inválido, ignorando")
    
    # 6. Soma
    print(f"\n📐 SOMA DAS DEZENAS (típico: 170-220)")
    soma_min = input(f"   Soma MÍNIMA [ENTER=pular]: ").strip()
    if soma_min:
        try:
            prefs['soma_min'] = int(soma_min)
        except ValueError:
            pass
    soma_max = input(f"   Soma MÁXIMA [ENTER=pular]: ").strip()
    if soma_max:
        try:
            prefs['soma_max'] = int(soma_max)
        except ValueError:
            pass
    
    # 7. Consecutivos
    print(f"\n📏 CONSECUTIVOS (típico: 3-7)")
    cons_input = input(f"   Máximo de consecutivos [ENTER=pular]: ").strip()
    if cons_input:
        try:
            prefs['max_consecutivos'] = int(cons_input)
        except ValueError:
            pass
    
    # 8. Dezenas fixas
    print(f"\n📌 DEZENAS FIXAS (OBRIGATÓRIAS)")
    print(f"   Digite as dezenas separadas por espaço")
    fixas_input = input(f"   Fixas [ENTER=pular]: ").strip()
    if fixas_input:
        try:
            fixas = [int(x) for x in fixas_input.split() if 1 <= int(x) <= 25]
            if fixas:
                prefs['fixas'] = sorted(set(fixas))[:15]
                print(f"   ✅ Fixas: {prefs['fixas']}")
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # 9. Dezenas excluídas
    print(f"\n🚫 DEZENAS EXCLUÍDAS (PROIBIDAS)")
    excl_input = input(f"   Excluídas [ENTER=pular]: ").strip()
    if excl_input:
        try:
            excl = [int(x) for x in excl_input.split() if 1 <= int(x) <= 25]
            if excl:
                # Não pode excluir as fixas
                if 'fixas' in prefs:
                    excl = [x for x in excl if x not in prefs['fixas']]
                prefs['excluidas'] = sorted(set(excl))
                print(f"   ✅ Excluídas: {prefs['excluidas']}")
        except ValueError:
            print(f"   ⚠️  Valor inválido, ignorando")
    
    # Mostrar resumo
    if prefs:
        print(f"\n📋 PREFERÊNCIAS CONFIGURADAS:")
        for key, value in prefs.items():
            if not key.endswith('_tolerance'):
                print(f"   {key}: {value}")
    else:
        print(f"\n📋 NENHUMA preferência → otimização puramente estrutural")
    
    return prefs


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def display_dashboard(data, profiles):
    """Exibe dashboard de navegação"""
    print(f"\n{'='*60}")
    print(f"🎯 DASHBOARD DE NAVEGAÇÃO ESTRUTURAL")
    print(f"{'='*60}")
    
    profile_counts = Counter(p['profile'] for p in profiles)
    print(f"\n📊 DISTRIBUIÇÃO DE PERFIS:")
    for profile, count in profile_counts.most_common():
        bar = '█' * (count * 40 // len(profiles))
        print(f"   {profile:<15} {bar} {count}")
    
    print(f"\n💡 PERFIS DISPONÍVEIS:")
    print(f"   conservador - Estrutura organizada, baixa variância")
    print(f"   caotico     - Máxima diversidade, alta independência")
    print(f"   cobertura   - Máxima cobertura de pares/trincas")
    print(f"   balanceado  - Equilíbrio entre todos os fatores")


def get_top_games_hybrid(data, profiles, profile_name='balanceado', 
                         preferences=None, ultimo_concurso=None,
                         top_pools=3, top_games=5):
    """
    Obtém os melhores jogos combinando:
    - Perfil estrutural (fronteira de Pareto)
    - Preferências humanas (score flexível)
    """
    # Filtrar pools pelo perfil
    matching = [(i, p) for i, p in enumerate(profiles) if p['profile'] == profile_name]
    
    if not matching:
        print(f"   ⚠️  Nenhum pool com perfil '{profile_name}'")
        return []
    
    matching.sort(key=lambda x: x[1]['scores'][profile_name], reverse=True)
    
    all_ranked = []
    
    for pool_idx, (i, p) in enumerate(matching[:top_pools]):
        pool = data['pareto_pools'][i]
        signals = p['signals']
        
        # Ranking híbrido
        ranked = hybrid_ranking(
            pool, signals, preferences or {}, ultimo_concurso,
            structural_weight=0.7, human_weight=0.3
        )
        
        # Adicionar identificação do pool
        for score, game, struct_s, human_s in ranked:
            all_ranked.append((score, game, struct_s, human_s, pool_idx, signals))
    
    # Reordenar globalmente
    all_ranked.sort(key=lambda x: x[0], reverse=True)
    
    return all_ranked[:top_games]


def display_top_games_hybrid(ranked_games, profile_name, preferences):
    """Exibe os melhores jogos com detalhes"""
    if not ranked_games:
        return
    
    print(f"\n{'='*60}")
    print(f"🏆 MELHORES JOGOS - PERFIL: {profile_name.upper()}")
    print(f"{'='*60}")
    
    if preferences:
        print(f"📋 Preferências ativas:")
        for key, value in preferences.items():
            if not key.endswith('_tolerance'):
                print(f"   {key}: {value}")
    
    for i, (final_score, game, struct_s, human_s, pool_idx, signals) in enumerate(ranked_games, 1):
        pares = sum(1 for d in game if d % 2 == 0)
        primos = sum(1 for d in game if d in PRIMES)
        moldura = sum(1 for d in game if d in MOLDURA)
        soma = sum(game)
        cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
        
        print(f"\n{'─'*50}")
        print(f"JOGO {i:02d} | Score: {final_score:.3f}")
        print(f"{'─'*50}")
        print(f"  Dezenas: {game}")
        print(f"  Estrutural: {struct_s:.3f} | Humano: {human_s:.3f}")
        print(f"  Pares: {pares} | Primos: {primos} | Moldura: {moldura}")
        print(f"  Soma: {soma} | Consecutivos: {cons}")
        print(f"  Pool: {pool_idx+1} | Perfil: {profile_name}")


def main():
    print("="*60)
    print("🧭 SISTEMA HÍBRIDO - ESTRUTURAL + PREFERÊNCIAS")
    print("="*60)
    
    # Carregar fronteira
    data = load_pareto_frontier()
    if data is None:
        return
    
    # Classificar perfis
    profiles = classify_profiles(data['signals'])
    
    # Extrair último concurso (simulado se não disponível)
    ultimo_concurso = data['pareto_pools'][0][0] if data['pareto_pools'] else None
    
    # Dashboard
    display_dashboard(data, profiles)
    
    # Loop interativo
    while True:
        print(f"\n{'='*60}")
        print(f"▶️  OPÇÕES:")
        print(f"   1. Escolher perfil estrutural")
        print(f"   2. Configurar preferências humanas")
        print(f"   3. Gerar jogos (combinando perfil + preferências)")
        print(f"   4. Sair")
        
        choice = input(f"\n   Opção: ").strip()
        
        if choice == '4':
            break
        
        elif choice == '1':
            print(f"\n   Perfis: conservador | caotico | cobertura | balanceado")
            profile = input(f"   Escolha: ").strip().lower()
            
            if profile in ['conservador', 'caotico', 'cobertura', 'balanceado']:
                current_profile = profile
                print(f"   ✅ Perfil: {profile}")
            else:
                print(f"   ⚠️  Perfil inválido")
        
        elif choice == '2':
            preferences = collect_user_preferences(ultimo_concurso)
            print(f"   ✅ {len(preferences)} preferências configuradas")
        
        elif choice == '3':
            if 'current_profile' not in dir():
                current_profile = 'balanceado'
                print(f"   ℹ️  Usando perfil padrão: balanceado")
            
            if 'preferences' not in dir():
                preferences = {}
            
            ranked = get_top_games_hybrid(
                data, profiles, current_profile,
                preferences, ultimo_concurso,
                top_pools=3, top_games=10
            )
            
            display_top_games_hybrid(ranked, current_profile, preferences)
    
    print(f"\n✅ ATÉ LOGO!")
    print(f"💡 Lembre-se: otimização estrutural + preferências humanas")
    print(f"   NÃO é previsão. É navegação inteligente no espaço combinatório.")


if __name__ == "__main__":
    main()
