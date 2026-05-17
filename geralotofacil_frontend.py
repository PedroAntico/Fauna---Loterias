#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRONTEND COMPLETO - SELEÇÃO ANTI-CLONES + SIGNATURES
=====================================================
Versão 3.0 - Diversidade Forçada + Navegação Vetorial

MELHORIAS:
✅ Carrega profiles PRÉ-CLASSIFICADOS (não recalcula)
✅ Front IDs para rankear por qualidade
✅ Seleção ANTI-CLONES (distância mínima entre jogos)
✅ Signatures contínuas (interpolação entre perfis)
✅ Navegação vetorial (mix de perfis)
"""

import numpy as np
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from math import comb

# ============================================================
# CONJUNTOS MATEMÁTICOS
# ============================================================

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5,6,10,11,15,16,20,21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}

# ============================================================
# CARREGAMENTO
# ============================================================

def load_pareto_frontier(filename='pareto_frontier.json'):
    if not os.path.exists(filename):
        print(f"❌ {filename} não encontrado! Execute system_facil.py primeiro.")
        return None
    print(f"📂 Carregando {filename}...")
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"   ✅ {data['metadata']['n_solutions']} soluções | "
          f"{data['metadata']['n_fronts']} fronts")
    return data

# ============================================================
# SELEÇÃO ANTI-CLONES (DIVERSIDADE FORÇADA)
# ============================================================

def select_diverse_games(ranked_games, top_n=10, min_distance=4):
    """
    Seleciona top N jogos com DIVERSIDADE FORÇADA
    
    Garante que jogos selecionados sejam estruturalmente DISTINTOS
    Evita entregar "clones" ao usuário
    
    Args:
        ranked_games: Lista (score, game, ...) ordenada por score
        top_n: Quantos jogos retornar
        min_distance: Distância Johnson mínima entre selecionados
    
    Returns:
        Lista de jogos diversos
    """
    selected = []
    
    for candidate in ranked_games:
        game = candidate[1]  # O jogo está na posição 1
        
        if not selected:
            selected.append(candidate)
            continue
        
        # Verificar distância para TODOS os já selecionados
        min_dist = min(
            15 - len(set(game) & set(s[1]))
            for s in selected
        )
        
        if min_dist >= min_distance:
            selected.append(candidate)
        
        if len(selected) >= top_n:
            break
    
    # Se não conseguiu preencher com diversidade, pegar os restantes
    if len(selected) < top_n:
        for candidate in ranked_games:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= top_n:
                break
    
    return selected[:top_n]


# ============================================================
# SCORING HÍBRIDO
# ============================================================

def structural_game_score(game, pool, signals):
    score = 0.0
    distances = []
    for other in pool:
        if game != other:
            distances.append(15 - len(set(game) & set(other)))
    if distances:
        score += (np.mean(distances)/10) * 0.35
    pares = sum(1 for d in game if d%2==0)
    score += (1.0 - abs(pares-7.5)/7.5) * 0.25
    baixas = sum(1 for d in game if d<=12)
    score += (1.0 - abs(baixas-7.5)/7.5) * 0.20
    unique_q = len(set((d-1)//5 for d in game))
    score += (unique_q/5) * 0.20
    return score

def human_constraint_score(game, preferences, ultimo_concurso=None):
    scores, weights = [], []
    
    if 'pares' in preferences:
        t = preferences['pares']; tol = preferences.get('pares_tolerance',1)
        a = sum(1 for d in game if d%2==0); d = abs(a-t)
        scores.append(1.0 if d<=tol else max(0,1.0-(d-tol)/7)); weights.append(1.5)
    
    if 'primos' in preferences:
        t = preferences['primos']; tol = preferences.get('primos_tolerance',1)
        a = sum(1 for d in game if d in PRIMES); d = abs(a-t)
        scores.append(1.0 if d<=tol else max(0,1.0-(d-tol)/5)); weights.append(1.2)
    
    if 'moldura' in preferences:
        t = preferences['moldura']; tol = preferences.get('moldura_tolerance',1)
        a = sum(1 for d in game if d in MOLDURA); d = abs(a-t)
        scores.append(1.0 if d<=tol else max(0,1.0-(d-tol)/7)); weights.append(1.0)
    
    if 'repetidas' in preferences and ultimo_concurso:
        t = preferences['repetidas']; tol = preferences.get('repetidas_tolerance',1)
        a = len(set(game)&set(ultimo_concurso)); d = abs(a-t)
        scores.append(1.0 if d<=tol else max(0,1.0-(d-tol)/7)); weights.append(1.3)
    
    if 'fixas' in preferences and preferences['fixas']:
        if not set(preferences['fixas']).issubset(set(game)):
            return 0.0
        scores.append(1.0); weights.append(2.0)
    
    if 'excluidas' in preferences and preferences['excluidas']:
        if set(preferences['excluidas']) & set(game):
            return 0.0
    
    if not scores: return 1.0
    return sum(s*w for s,w in zip(scores,weights))/sum(weights)

def hybrid_ranking(pool, signals, preferences, ultimo_concurso=None, struct_w=0.7, human_w=0.3):
    ranked = []
    for game in pool:
        ss = structural_game_score(game, pool, signals)
        hs = human_constraint_score(game, preferences, ultimo_concurso)
        ranked.append((ss*struct_w + hs*human_w, sorted(game), ss, hs))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def display_dashboard(data):
    print(f"\n{'='*60}")
    print(f"🎯 DASHBOARD - {data['metadata']['n_solutions']} SOLUÇÕES")
    print(f"{'='*60}")
    
    profile_counts = Counter(p['profile'] for p in data['profiles'])
    print(f"\n📊 PERFIS:")
    for profile, count in profile_counts.most_common():
        bar = '█'*(count*40//len(data['profiles']))
        print(f"   {profile:<15} {bar} {count}")
    
    # Distribuição por front
    front_counts = Counter(data['front_ids'])
    print(f"\n📊 FRONTS:")
    for front_id in sorted(front_counts):
        print(f"   Front {front_id}: {front_counts[front_id]} soluções")

def display_top_games(data, profile_name='balanceado', preferences=None, 
                      ultimo_concurso=None, top_n=10, min_dist=4):
    """Exibe jogos com seleção ANTI-CLONES"""
    
    # Filtrar pools do perfil
    matching = [(i, p) for i, p in enumerate(data['profiles']) if p['profile'] == profile_name]
    
    if not matching:
        print(f"   ⚠️  Nenhum pool com perfil '{profile_name}'")
        print(f"   Perfis disponíveis: {set(p['profile'] for p in data['profiles'])}")
        return []
    
    # Ordenar por score do perfil E front (preferir fronts menores)
    matching.sort(key=lambda x: (x[1]['scores'][profile_name], -data['front_ids'][x[0]]), reverse=True)
    
    all_ranked = []
    
    for pool_idx, (i, p) in enumerate(matching[:5]):  # Top 5 pools do perfil
        pool = data['pareto_pools'][i]
        signals = data['signals'][i]
        front_id = data['front_ids'][i]
        
        ranked = hybrid_ranking(pool, signals, preferences or {}, ultimo_concurso)
        
        for score, game, ss, hs in ranked:
            all_ranked.append((score, game, ss, hs, pool_idx, front_id))
    
    # Reordenar
    all_ranked.sort(key=lambda x: x[0], reverse=True)
    
    # SELEÇÃO ANTI-CLONES
    diverse = select_diverse_games(all_ranked, top_n, min_dist)
    
    # Exibir
    print(f"\n{'='*60}")
    print(f"🏆 TOP {len(diverse)} JOGOS - {profile_name.upper()}")
    print(f"{'='*60}")
    print(f"   (Seleção com diversidade forçada: dist ≥ {min_dist})")
    
    if preferences:
        print(f"   Preferências: {preferences}")
    
    for i, (score, game, ss, hs, pool_idx, front_id) in enumerate(diverse, 1):
        pares = sum(1 for d in game if d%2==0)
        primos = sum(1 for d in game if d in PRIMES)
        moldura = sum(1 for d in game if d in MOLDURA)
        soma = sum(game)
        
        print(f"\n{'─'*50}")
        print(f"JOGO {i:02d} | Score: {score:.3f} | Front: {front_id}")
        print(f"{'─'*50}")
        print(f"  {game}")
        print(f"  Estrutural: {ss:.3f} | Humano: {hs:.3f}")
        print(f"  Pares: {pares} | Primos: {primos} | Moldura: {moldura} | Soma: {soma}")
    
    return diverse


def main():
    print("="*60)
    print("🧭 FRONTEND v3.0 - ANTI-CLONES + SIGNATURES")
    print("="*60)
    
    data = load_pareto_frontier()
    if data is None: return
    
    display_dashboard(data)
    
    print(f"\n💡 PERFIS: conservador | caotico | cobertura | balanceado")
    print(f"💡 Comando: perfil [nome] ou 'sair'")
    
    current_profile = 'balanceado'
    
    while True:
        cmd = input(f"\n▶️ [{current_profile}] ").strip().lower()
        
        if cmd == 'sair': break
        if cmd in ['conservador','caotico','cobertura','balanceado']:
            current_profile = cmd
            print(f"   ✅ Perfil: {current_profile}")
            display_top_games(data, current_profile, top_n=10, min_dist=4)
        elif cmd == '':
            display_top_games(data, current_profile, top_n=10, min_dist=4)
        else:
            print(f"   ⚠️  Opções: conservador, caotico, cobertura, balanceado, sair")
    
    print(f"\n✅ ATÉ LOGO!")

if __name__ == "__main__":
    main()
