#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRONTEND COMPLETO - PREFERÊNCIAS HUMANAS + OTIMIZAÇÃO
======================================================
Versão 2.0 - Interface Completa de Preferências

OPÇÕES:
✅ Pares (quantidade + tolerância)
✅ Primos (quantidade + tolerância)
✅ Moldura (quantidade + tolerância)
✅ Centro (quantidade + tolerância)
✅ Repetidas do último concurso
✅ Soma (faixa min-max)
✅ Consecutivos (máximo)
✅ Dezenas fixas (obrigatório)
✅ Dezenas excluídas (obrigatório)
✅ Peso estrutural vs humano (ajustável)
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

MOLDURA = {
    1, 2, 3, 4, 5,
    6, 10, 11, 15,
    16, 20, 21, 22, 23, 24, 25
}

CENTRO = {7, 8, 9, 12, 13, 14, 17, 18, 19}

QUADRANTES = {
    'Q1': {1, 2, 3, 4, 5},
    'Q2': {6, 7, 8, 9, 10},
    'Q3': {11, 12, 13, 14, 15},
    'Q4': {16, 17, 18, 19, 20},
    'Q5': {21, 22, 23, 24, 25}
}

# ============================================================
# CARREGAMENTO
# ============================================================

def load_pareto_frontier(filename='pareto_frontier.json'):
    if not os.path.exists(filename):
        print(f"❌ Arquivo {filename} não encontrado!")
        return None
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📂 {data['metadata']['n_solutions']} soluções carregadas")
    return data


# ============================================================
# CLASSIFICAÇÃO DE PERFIS
# ============================================================

def classify_profiles(signals_list):
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
            'conservador': (1 - norm.get('pos_entropy', 0.5)) * 1.5 + norm.get('compressibility', 0.5) * 1.5 + (1 - norm.get('johnson_avg', 0.5)) * 1.0,
            'caotico': norm.get('johnson_min', 0.5) * 1.5 + norm.get('johnson_avg', 0.5) * 1.5 + norm.get('pos_entropy', 0.5) * 1.0,
            'cobertura': norm.get('pair_coverage', 0.5) * 1.5 + (1 - norm.get('covering_radius', 0.5)) * 1.5 + norm.get('sphere_packing', 0.5) * 1.0,
            'balanceado': norm.get('pair_coverage', 0.5) * 1.0 + norm.get('johnson_min', 0.5) * 1.0 + (1 - norm.get('covering_radius', 0.5)) * 1.0 + norm.get('pos_entropy', 0.5) * 1.0
        }
        profiles.append({'profile': max(scores, key=scores.get), 'scores': scores, 'signals': signals})
    return profiles


# ============================================================
# SCORE ESTRUTURAL
# ============================================================

def structural_game_score(game, pool, signals):
    score = 0.0
    distances = []
    for other in pool:
        if game != other:
            common = len(set(game) & set(other))
            distances.append(15 - common)
    if distances:
        score += (np.mean(distances) / 10) * 0.35
    pares = sum(1 for d in game if d % 2 == 0)
    score += (1.0 - abs(pares - 7.5) / 7.5) * 0.25
    baixas = sum(1 for d in game if d <= 12)
    score += (1.0 - abs(baixas - 7.5) / 7.5) * 0.20
    unique_q = len(set((d-1)//5 for d in game))
    score += (unique_q / 5) * 0.20
    return score


# ============================================================
# SCORE HUMANO (FLEXÍVEL)
# ============================================================

def human_constraint_score(game, preferences, ultimo_concurso=None):
    scores = []
    weights = []
    
    if 'pares' in preferences:
        target = preferences['pares']
        tol = preferences.get('pares_tolerance', 1)
        actual = sum(1 for d in game if d % 2 == 0)
        dist = abs(actual - target)
        scores.append(1.0 if dist <= tol else max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.5)
    
    if 'primos' in preferences:
        target = preferences['primos']
        tol = preferences.get('primos_tolerance', 1)
        actual = sum(1 for d in game if d in PRIMES)
        dist = abs(actual - target)
        scores.append(1.0 if dist <= tol else max(0, 1.0 - (dist - tol) / 5))
        weights.append(1.2)
    
    if 'moldura' in preferences:
        target = preferences['moldura']
        tol = preferences.get('moldura_tolerance', 1)
        actual = sum(1 for d in game if d in MOLDURA)
        dist = abs(actual - target)
        scores.append(1.0 if dist <= tol else max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.0)
    
    if 'centro' in preferences:
        target = preferences['centro']
        tol = preferences.get('centro_tolerance', 1)
        actual = sum(1 for d in game if d in CENTRO)
        dist = abs(actual - target)
        scores.append(1.0 if dist <= tol else max(0, 1.0 - (dist - tol) / 5))
        weights.append(0.8)
    
    if 'repetidas' in preferences and ultimo_concurso:
        target = preferences['repetidas']
        tol = preferences.get('repetidas_tolerance', 1)
        actual = len(set(game) & set(ultimo_concurso))
        dist = abs(actual - target)
        scores.append(1.0 if dist <= tol else max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.3)
    
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
    
    if 'max_consecutivos' in preferences:
        max_cons = preferences['max_consecutivos']
        d = sorted(game)
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
        scores.append(1.0 if cons <= max_cons else max(0, 1.0 - (cons - max_cons) / 5))
        weights.append(0.9)
    
    if 'fixas' in preferences and preferences['fixas']:
        if not set(preferences['fixas']).issubset(set(game)):
            return 0.0
        scores.append(1.0)
        weights.append(2.0)
    
    if 'excluidas' in preferences and preferences['excluidas']:
        if set(preferences['excluidas']) & set(game):
            return 0.0
    
    if not scores:
        return 1.0
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


# ============================================================
# RANKEAMENTO HÍBRIDO
# ============================================================

def hybrid_ranking(pool, signals, preferences, ultimo_concurso=None, structural_weight=0.7):
    ranked = []
    for game in pool:
        struct_score = structural_game_score(game, pool, signals)
        human_score = human_constraint_score(game, preferences, ultimo_concurso)
        final_score = struct_score * structural_weight + human_score * (1 - structural_weight)
        ranked.append((final_score, sorted(game), struct_score, human_score))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ============================================================
# COLETA DE PREFERÊNCIAS
# ============================================================

def collect_preferences(ultimo_concurso=None):
    print(f"\n{'='*60}")
    print(f"🎯 CONFIGURAÇÃO DE PREFERÊNCIAS")
    print(f"{'='*60}")
    print(f"💡 ENTER = sem restrição | Valores típicos entre parênteses")
    
    prefs = {}
    
    # Pares
    print(f"\n📊 PARES (típico: 6-9)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['pares'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['pares_tolerance'] = int(t) if t else 1
        except: pass
    
    # Primos
    print(f"\n🔢 PRIMOS: {sorted(PRIMES)} (típico: 3-6)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['primos'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['primos_tolerance'] = int(t) if t else 1
        except: pass
    
    # Moldura
    print(f"\n🖼️  MOLDURA (típico: 7-10)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['moldura'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['moldura_tolerance'] = int(t) if t else 1
        except: pass
    
    # Centro
    print(f"\n🎯 CENTRO (típico: 5-8)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['centro'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['centro_tolerance'] = int(t) if t else 1
        except: pass
    
    # Repetidas
    if ultimo_concurso:
        print(f"\n🔄 REPETIDAS DO ÚLTIMO CONCURSO")
        print(f"   Último: {sorted(ultimo_concurso)}")
        v = input(f"   Quantidade [ENTER=pular]: ").strip()
        if v:
            try:
                prefs['repetidas'] = int(v)
                t = input(f"   Tolerância (±) [1]: ").strip()
                prefs['repetidas_tolerance'] = int(t) if t else 1
            except: pass
    
    # Soma
    print(f"\n📐 SOMA (típico: 170-220)")
    v = input(f"   Mínima [ENTER=pular]: ").strip()
    if v:
        try: prefs['soma_min'] = int(v)
        except: pass
    v = input(f"   Máxima [ENTER=pular]: ").strip()
    if v:
        try: prefs['soma_max'] = int(v)
        except: pass
    
    # Consecutivos
    print(f"\n📏 CONSECUTIVOS (típico: 3-7)")
    v = input(f"   Máximo [ENTER=pular]: ").strip()
    if v:
        try: prefs['max_consecutivos'] = int(v)
        except: pass
    
    # Fixas
    print(f"\n📌 DEZENAS FIXAS (separadas por espaço)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            fixas = sorted(set(int(x) for x in v.split() if 1 <= int(x) <= 25))
            if fixas: prefs['fixas'] = fixas[:15]
        except: pass
    
    # Excluídas
    print(f"\n🚫 DEZENAS EXCLUÍDAS (separadas por espaço)")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            excl = [int(x) for x in v.split() if 1 <= int(x) <= 25]
            if 'fixas' in prefs:
                excl = [x for x in excl if x not in prefs['fixas']]
            if excl: prefs['excluidas'] = sorted(set(excl))
        except: pass
    
    # Peso estrutural
    print(f"\n⚖️  PESO ESTRUTURAL (0-1)")
    print(f"   1.0 = só otimização | 0.0 = só preferências")
    v = input(f"   Peso [0.7]: ").strip()
    if v:
        try: prefs['structural_weight'] = float(v)
        except: pass
    
    # Resumo
    if prefs:
        print(f"\n📋 PREFERÊNCIAS:")
        for k, v in prefs.items():
            if not k.endswith('_tolerance'):
                print(f"   {k}: {v}")
    else:
        print(f"\n📋 NENHUMA preferência → otimização pura")
    
    return prefs


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def display_dashboard(data, profiles):
    print(f"\n{'='*60}")
    print(f"🎯 DASHBOARD")
    print(f"{'='*60}")
    counts = Counter(p['profile'] for p in profiles)
    for profile, count in counts.most_common():
        bar = '█' * (count * 40 // len(profiles))
        print(f"   {profile:<15} {bar} {count}")


def get_top_games(data, profiles, profile_name, preferences, ultimo_concurso, top_pools=3, top_games=10):
    matching = [(i, p) for i, p in enumerate(profiles) if p['profile'] == profile_name]
    if not matching:
        print(f"   ⚠️  Perfil '{profile_name}' não encontrado")
        return []
    
    matching.sort(key=lambda x: x[1]['scores'][profile_name], reverse=True)
    
    structural_weight = preferences.get('structural_weight', 0.7) if preferences else 0.7
    
    all_ranked = []
    for i, p in matching[:top_pools]:
        pool = data['pareto_pools'][i]
        ranked = hybrid_ranking(pool, p['signals'], preferences or {}, ultimo_concurso, structural_weight)
        for score, game, ss, hs in ranked:
            all_ranked.append((score, game, ss, hs, i, p['signals']))
    
    all_ranked.sort(key=lambda x: x[0], reverse=True)
    return all_ranked[:top_games]


def display_top_games(ranked_games, profile_name, preferences):
    if not ranked_games:
        return
    
    print(f"\n{'='*60}")
    print(f"🏆 TOP JOGOS - {profile_name.upper()}")
    print(f"{'='*60}")
    
    for i, (score, game, ss, hs, pool_idx, signals) in enumerate(ranked_games, 1):
        pares = sum(1 for d in game if d % 2 == 0)
        primos = sum(1 for d in game if d in PRIMES)
        moldura = sum(1 for d in game if d in MOLDURA)
        soma = sum(game)
        cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
        
        print(f"\n{'─'*50}")
        print(f"JOGO {i:02d} | Score: {score:.3f} (E:{ss:.2f} H:{hs:.2f})")
        print(f"{'─'*50}")
        print(f"  {game}")
        print(f"  Pares:{pares} | Primos:{primos} | Moldura:{moldura} | Soma:{soma} | Cons:{cons}")


def main():
    print("="*60)
    print("🧭 FRONTEND COMPLETO - PREFERÊNCIAS + ESTRUTURAL")
    print("="*60)
    
    data = load_pareto_frontier()
    if data is None:
        return
    
    profiles = classify_profiles(data['signals'])
    ultimo = data['pareto_pools'][0][0] if data['pareto_pools'] else None
    
    display_dashboard(data, profiles)
    
    current_profile = 'balanceado'
    preferences = {}
    
    while True:
        print(f"\n{'='*60}")
        print(f"▶️  Perfil: {current_profile} | Prefs: {len(preferences)}")
        print(f"   1. Mudar perfil (conservador/caotico/cobertura/balanceado)")
        print(f"   2. Configurar preferências (pares, primos, moldura...)")
        print(f"   3. GERAR JOGOS")
        print(f"   4. Sair")
        
        c = input(f"\n   Opção: ").strip()
        
        if c == '4': break
        elif c == '1':
            p = input(f"   Perfil: ").strip().lower()
            if p in ['conservador', 'caotico', 'cobertura', 'balanceado']:
                current_profile = p
        elif c == '2':
            preferences = collect_preferences(ultimo)
        elif c == '3':
            ranked = get_top_games(data, profiles, current_profile, preferences, ultimo)
            display_top_games(ranked, current_profile, preferences)
    
    print(f"\n✅ ATÉ LOGO!")


if __name__ == "__main__":
    main()
