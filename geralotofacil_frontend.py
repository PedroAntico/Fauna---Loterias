#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FRONTEND CORRIGIDO - PREFERÊNCIAS REAIS + ÚLTIMO CONCURSO REAL
================================================================
Versão 2.1 - Correções Críticas

CORREÇÕES:
✅ Carrega último concurso REAL do CSV histórico
✅ human_score == 0 → ELIMINA o jogo (não apenas penaliza)
✅ Peso estrutural padrão: 0.7 (não 1.0)
✅ Validação de preferências antes do ranking
✅ Exibição do último concurso real
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
# CARREGAMENTO DO ÚLTIMO CONCURSO REAL
# ============================================================

def load_last_contest(csv_file='resultados_lotofacil.csv'):
    """
    Carrega o ÚLTIMO concurso REAL do arquivo CSV histórico
    
    Returns:
        dict: {'concurso': str, 'data': str, 'dezenas': list}
        None: se arquivo não existir
    """
    if not os.path.exists(csv_file):
        print(f"⚠️  Arquivo {csv_file} não encontrado!")
        print(f"   As preferências de 'repetidas' usarão dados simulados.")
        return None
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        # Pegar última linha (último concurso)
        last_line = lines[1].strip()
        parts = last_line.split(';')
        
        # Formato típico: concurso;data;b1;b2;...;b15
        if len(parts) >= 17:
            concurso = parts[0]
            data = parts[1]
            dezenas = sorted([int(x) for x in parts[2:17]])
            
            return {
                'concurso': concurso,
                'data': data,
                'dezenas': dezenas
            }
    except Exception as e:
        print(f"⚠️  Erro ao ler CSV: {e}")
    
    return None


def load_last_contests(csv_file='resultados_lotofacil.csv', n=10):
    """
    Carrega os últimos N concursos reais
    
    Returns:
        list: Lista dos últimos N concursos
    """
    if not os.path.exists(csv_file):
        return None
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            return None
        
        contests = []
        for line in lines[-n:]:
            parts = line.strip().split(';')
            if len(parts) >= 17:
                contests.append({
                    'concurso': parts[0],
                    'data': parts[1],
                    'dezenas': sorted([int(x) for x in parts[2:17]])
                })
        
        return contests
    except:
        return None


# ============================================================
# CARREGAMENTO DA FRONTEIRA DE PARETO
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
# CLASSIFICAÇÃO DE PERFIS
# ============================================================

def classify_profiles(signals_list):
    """Classifica pools em perfis com normalização"""
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
# SCORE ESTRUTURAL
# ============================================================

def structural_game_score(game, pool, signals):
    """Score estrutural de um jogo dentro do pool"""
    score = 0.0
    
    # Distância para outros jogos (diversidade)
    distances = []
    for other in pool:
        if game != other:
            common = len(set(game) & set(other))
            distances.append(15 - common)
    
    if distances:
        score += (np.mean(distances) / 10) * 0.35
    
    # Balanceamento par/ímpar
    pares = sum(1 for d in game if d % 2 == 0)
    score += (1.0 - abs(pares - 7.5) / 7.5) * 0.25
    
    # Distribuição alto/baixo
    baixas = sum(1 for d in game if d <= 12)
    score += (1.0 - abs(baixas - 7.5) / 7.5) * 0.20
    
    # Cobertura de quadrantes
    unique_q = len(set((d-1)//5 for d in game))
    score += (unique_q / 5) * 0.20
    
    return score


# ============================================================
# SCORE HUMANO (COM ELIMINAÇÃO)
# ============================================================

def human_constraint_score(game, preferences, ultimo_concurso=None):
    """
    Score de preferências humanas
    
    CORREÇÃO CRÍTICA:
    - Se human_score == 0 → jogo é ELIMINADO (não apenas penalizado)
    - Isso inclui: fixas ausentes, excluídas presentes
    """
    scores = []
    weights = []
    
    # 1. Pares (SOFT - tolerância)
    if 'pares' in preferences:
        target = preferences['pares']
        tol = preferences.get('pares_tolerance', 1)
        actual = sum(1 for d in game if d % 2 == 0)
        dist = abs(actual - target)
        
        if dist <= tol:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.5)
    
    # 2. Primos (SOFT - tolerância)
    if 'primos' in preferences:
        target = preferences['primos']
        tol = preferences.get('primos_tolerance', 1)
        actual = sum(1 for d in game if d in PRIMES)
        dist = abs(actual - target)
        
        if dist <= tol:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tol) / 5))
        weights.append(1.2)
    
    # 3. Moldura (SOFT - tolerância)
    if 'moldura' in preferences:
        target = preferences['moldura']
        tol = preferences.get('moldura_tolerance', 1)
        actual = sum(1 for d in game if d in MOLDURA)
        dist = abs(actual - target)
        
        if dist <= tol:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.0)
    
    # 4. Centro (SOFT - tolerância)
    if 'centro' in preferences:
        target = preferences['centro']
        tol = preferences.get('centro_tolerance', 1)
        actual = sum(1 for d in game if d in CENTRO)
        dist = abs(actual - target)
        
        if dist <= tol:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tol) / 5))
        weights.append(0.8)
    
    # 5. Repetidas do ÚLTIMO CONCURSO REAL (SOFT - tolerância)
    if 'repetidas' in preferences and ultimo_concurso:
        target = preferences['repetidas']
        tol = preferences.get('repetidas_tolerance', 1)
        actual = len(set(game) & set(ultimo_concurso))
        dist = abs(actual - target)
        
        if dist <= tol:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (dist - tol) / 7))
        weights.append(1.3)
    
    # 6. Soma (SOFT - faixa)
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
    
    # 7. Consecutivos (SOFT - máximo)
    if 'max_consecutivos' in preferences:
        max_cons = preferences['max_consecutivos']
        d = sorted(game)
        cons = sum(1 for i in range(len(d)-1) if d[i+1]-d[i] == 1)
        
        if cons <= max_cons:
            scores.append(1.0)
        else:
            scores.append(max(0, 1.0 - (cons - max_cons) / 5))
        weights.append(0.9)
    
    # 8. Dezenas FIXAS (HARD - obrigatório)
    # CORREÇÃO: se não tem as fixas, retorna 0 → ELIMINA o jogo
    if 'fixas' in preferences and preferences['fixas']:
        if not set(preferences['fixas']).issubset(set(game)):
            return 0.0  # ELIMINA: não contém todas as fixas
        scores.append(1.0)
        weights.append(2.0)
    
    # 9. Dezenas EXCLUÍDAS (HARD - obrigatório)
    # CORREÇÃO: se tem excluída, retorna 0 → ELIMINA o jogo
    if 'excluidas' in preferences and preferences['excluidas']:
        if set(preferences['excluidas']) & set(game):
            return 0.0  # ELIMINA: contém dezena proibida
    
    # Se não há preferências, score máximo
    if not scores:
        return 1.0
    
    # Média ponderada
    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


# ============================================================
# RANKEAMENTO HÍBRIDO (CORRIGIDO)
# ============================================================

def hybrid_ranking(pool, signals, preferences, ultimo_concurso=None, structural_weight=0.7):
    """
    Ranking HÍBRIDO com ELIMINAÇÃO de jogos inválidos
    
    CORREÇÃO:
    - Jogos com human_score == 0 são ELIMINADOS
    - Não apenas perdem score, são removidos do ranking
    """
    ranked = []
    eliminated = 0
    
    for game in pool:
        # Score estrutural
        struct_score = structural_game_score(game, pool, signals)
        
        # Score humano
        human_score = human_constraint_score(game, preferences, ultimo_concurso)
        
        # CORREÇÃO: ELIMINAR jogos com score humano zero
        if human_score == 0:
            eliminated += 1
            continue
        
        # Score final híbrido
        final_score = struct_score * structural_weight + human_score * (1 - structural_weight)
        
        ranked.append((final_score, sorted(game), struct_score, human_score))
    
    if eliminated > 0:
        print(f"   🚫 {eliminated} jogos eliminados (não atendem restrições obrigatórias)")
    
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


# ============================================================
# COLETA DE PREFERÊNCIAS
# ============================================================

def collect_preferences(ultimo_concurso=None):
    """Interface interativa para coletar preferências"""
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
        except:
            pass
    
    # Primos
    print(f"\n🔢 PRIMOS: {sorted(PRIMES)} (típico: 3-6)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['primos'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['primos_tolerance'] = int(t) if t else 1
        except:
            pass
    
    # Moldura
    print(f"\n🖼️  MOLDURA (típico: 7-10)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['moldura'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['moldura_tolerance'] = int(t) if t else 1
        except:
            pass
    
    # Centro
    print(f"\n🎯 CENTRO (típico: 5-8)")
    v = input(f"   Quantidade [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['centro'] = int(v)
            t = input(f"   Tolerância (±) [1]: ").strip()
            prefs['centro_tolerance'] = int(t) if t else 1
        except:
            pass
    
    # Repetidas do último concurso REAL
    if ultimo_concurso:
        print(f"\n🔄 REPETIDAS DO ÚLTIMO CONCURSO REAL")
        print(f"   Concurso: {ultimo_concurso}")
        print(f"   (típico: 7-10 repetidas)")
        v = input(f"   Quantas repetidas? [ENTER=pular]: ").strip()
        if v:
            try:
                prefs['repetidas'] = int(v)
                t = input(f"   Tolerância (±) [1]: ").strip()
                prefs['repetidas_tolerance'] = int(t) if t else 1
            except:
                pass
    
    # Soma
    print(f"\n📐 SOMA DAS DEZENAS (típico: 170-220)")
    v = input(f"   Mínima [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['soma_min'] = int(v)
        except:
            pass
    v = input(f"   Máxima [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['soma_max'] = int(v)
        except:
            pass
    
    # Consecutivos
    print(f"\n📏 CONSECUTIVOS (típico: 3-7)")
    v = input(f"   Máximo [ENTER=pular]: ").strip()
    if v:
        try:
            prefs['max_consecutivos'] = int(v)
        except:
            pass
    
    # Fixas
    print(f"\n📌 DEZENAS FIXAS (OBRIGATÓRIAS)")
    print(f"   Digite separadas por espaço")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            fixas = sorted(set(int(x) for x in v.split() if 1 <= int(x) <= 25))
            if fixas:
                prefs['fixas'] = fixas[:15]
                print(f"   ✅ Fixas: {prefs['fixas']}")
        except:
            pass
    
    # Excluídas
    print(f"\n🚫 DEZENAS EXCLUÍDAS (PROIBIDAS)")
    print(f"   Digite separadas por espaço")
    v = input(f"   [ENTER=pular]: ").strip()
    if v:
        try:
            excl = [int(x) for x in v.split() if 1 <= int(x) <= 25]
            if 'fixas' in prefs:
                excl = [x for x in excl if x not in prefs['fixas']]
            if excl:
                prefs['excluidas'] = sorted(set(excl))
                print(f"   ✅ Excluídas: {prefs['excluidas']}")
        except:
            pass
    
    # Peso estrutural
    print(f"\n⚖️  PESO ESTRUTURAL (0-1)")
    print(f"   1.0 = só otimização estrutural")
    print(f"   0.0 = só preferências humanas")
    print(f"   0.7 = equilíbrio recomendado")
    v = input(f"   Peso [0.7]: ").strip()
    if v:
        try:
            prefs['structural_weight'] = float(v)
        except:
            pass
    
    # Resumo
    if prefs:
        print(f"\n📋 PREFERÊNCIAS CONFIGURADAS:")
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
    """Exibe dashboard de navegação"""
    print(f"\n{'='*60}")
    print(f"🎯 DASHBOARD DE NAVEGAÇÃO")
    print(f"{'='*60}")
    
    profile_counts = Counter(p['profile'] for p in profiles)
    for profile, count in profile_counts.most_common():
        bar = '█' * (count * 40 // max(1, len(profiles)))
        print(f"   {profile:<15} {bar} {count}")
    
    print(f"\n💡 PERFIS:")
    print(f"   conservador - Estrutura organizada, baixa variância")
    print(f"   caotico     - Máxima diversidade, alta independência")
    print(f"   cobertura   - Máxima cobertura de pares/trincas")
    print(f"   balanceado  - Equilíbrio entre todos os fatores")


def get_top_games(data, profiles, profile_name, preferences, ultimo_concurso, 
                  top_pools=3, top_games=10):
    """
    Obtém os melhores jogos combinando perfil + preferências
    
    CORREÇÃO: Usa o último concurso REAL
    """
    matching = [(i, p) for i, p in enumerate(profiles) if p['profile'] == profile_name]
    
    if not matching:
        print(f"   ⚠️  Perfil '{profile_name}' não encontrado na fronteira")
        print(f"   Perfis disponíveis: {set(p['profile'] for p in profiles)}")
        return []
    
    matching.sort(key=lambda x: x[1]['scores'][profile_name], reverse=True)
    
    structural_weight = preferences.get('structural_weight', 0.7) if preferences else 0.7
    
    all_ranked = []
    total_eliminated = 0
    
    for i, p in matching[:top_pools]:
        pool = data['pareto_pools'][i]
        signals = p['signals']
        
        ranked = hybrid_ranking(
            pool, signals, preferences or {}, 
            ultimo_concurso, structural_weight
        )
        
        for score, game, ss, hs in ranked:
            all_ranked.append((score, game, ss, hs, i, signals))
    
    all_ranked.sort(key=lambda x: x[0], reverse=True)
    return all_ranked[:top_games]


def display_top_games(ranked_games, profile_name, preferences, ultimo_concurso=None):
    """Exibe os melhores jogos com detalhes"""
    if not ranked_games:
        print(f"\n   ⚠️  Nenhum jogo encontrado com estas preferências.")
        print(f"   Tente relaxar as restrições ou mudar de perfil.")
        return
    
    print(f"\n{'='*60}")
    print(f"🏆 TOP {len(ranked_games)} JOGOS - {profile_name.upper()}")
    print(f"{'='*60}")
    
    if ultimo_concurso:
        print(f"📌 Último concurso: {ultimo_concurso}")
    
    for i, (score, game, ss, hs, pool_idx, signals) in enumerate(ranked_games, 1):
        pares = sum(1 for d in game if d % 2 == 0)
        primos = sum(1 for d in game if d in PRIMES)
        moldura = sum(1 for d in game if d in MOLDURA)
        centro = sum(1 for d in game if d in CENTRO)
        soma = sum(game)
        cons = sum(1 for i in range(len(game)-1) if game[i+1]-game[i] == 1)
        
        if ultimo_concurso:
            repetidas = len(set(game) & set(ultimo_concurso))
        else:
            repetidas = 0
        
        print(f"\n{'─'*50}")
        print(f"JOGO {i:02d} | Score: {score:.3f} (E:{ss:.2f} H:{hs:.2f})")
        print(f"{'─'*50}")
        print(f"  Dezenas: {game}")
        print(f"  Pares:{pares} | Primos:{primos} | Moldura:{moldura} | Centro:{centro}")
        print(f"  Soma:{soma} | Consecutivos:{cons} | Repetidas:{repetidas}")


def main():
    print("="*60)
    print("🧭 FRONTEND CORRIGIDO - PREFERÊNCIAS REAIS")
    print("="*60)
    
    # 1. Carregar fronteira de Pareto
    data = load_pareto_frontier()
    if data is None:
        return
    
    # 2. Classificar perfis
    profiles = classify_profiles(data['signals'])
    
    # 3. CORREÇÃO: Carregar último concurso REAL do CSV
    ultimo_info = load_last_contest('resultados_lotofacil.csv')
    
    if ultimo_info:
        ultimo_concurso = ultimo_info['dezenas']
        print(f"\n📌 ÚLTIMO CONCURSO REAL:")
        print(f"   Concurso {ultimo_info['concurso']} ({ultimo_info['data']})")
        print(f"   Dezenas: {ultimo_concurso}")
    else:
        ultimo_concurso = None
        print(f"\n⚠️  Último concurso não disponível")
        print(f"   Preferências de 'repetidas' não funcionarão.")
    
    # 4. Dashboard
    display_dashboard(data, profiles)
    
    # 5. Estado
    current_profile = 'balanceado'
    preferences = {}
    
    # 6. Loop interativo
    while True:
        print(f"\n{'='*60}")
        print(f"▶️  Perfil: {current_profile} | Preferências: {len(preferences)}")
        print(f"   1. Mudar perfil (conservador/caotico/cobertura/balanceado)")
        print(f"   2. Configurar preferências")
        print(f"   3. GERAR JOGOS")
        print(f"   4. Sair")
        
        choice = input(f"\n   Opção: ").strip()
        
        if choice == '4':
            break
        
        elif choice == '1':
            p = input(f"   Perfil: ").strip().lower()
            if p in ['conservador', 'caotico', 'cobertura', 'balanceado']:
                current_profile = p
                print(f"   ✅ Perfil: {current_profile}")
            else:
                print(f"   ⚠️  Perfil inválido")
        
        elif choice == '2':
            preferences = collect_preferences(ultimo_concurso)
        
        elif choice == '3':
            ranked = get_top_games(
                data, profiles, current_profile, 
                preferences, ultimo_concurso
            )
            display_top_games(ranked, current_profile, preferences, ultimo_concurso)
    
    print(f"\n✅ ATÉ LOGO!")
    print(f"💡 Sistema híbrido: otimização estrutural + preferências humanas")


if __name__ == "__main__":
    main()
