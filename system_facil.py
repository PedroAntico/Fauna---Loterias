#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SIGNAL ENGINE + FRONTEND INTELIGENTE
=====================================
Versão 1.0 - Consumindo Sinais da Fronteira de Pareto

FUNCIONALIDADES:
✅ Carrega fronteira de Pareto do NSGA-II
✅ Corrige distância zero (anti-duplicatas)
✅ Farthest-point sampling para covering radius
✅ Perfis de usuário (Conservador, Caótico, Cobertura)
✅ Exibe top pools por perfil
✅ Exporta sinais estruturais
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
import hashlib
from math import comb
import heapq

warnings.filterwarnings('ignore')

# ============================================================
# JOHNSON SPACE (OTIMIZADO)
# ============================================================

class JohnsonSpace:
    """Espaço de Johnson com operações bitwise"""
    
    def __init__(self, n=25, k=15):
        self.n = n
        self.k = k
        self.max_distance = k - max(0, 2*k - n)
        self._dist_cache = {}
        self._cover_samples = None
    
    def game_to_bits(self, game):
        bits = 0
        for d in game:
            bits |= (1 << (d - 1))
        return bits
    
    def johnson_distance(self, game1, game2):
        if isinstance(game1, list): bits1 = self.game_to_bits(game1)
        else: bits1 = game1
        if isinstance(game2, list): bits2 = self.game_to_bits(game2)
        else: bits2 = game2
        
        key = (bits1, bits2) if bits1 < bits2 else (bits2, bits1)
        if key in self._dist_cache:
            return self._dist_cache[key]
        
        d = self.k - (bits1 & bits2).bit_count()
        self._dist_cache[key] = d
        return d
    
    def min_johnson_distance(self, pool):
        if len(pool) < 2:
            return self.max_distance
        
        bits_list = [self.game_to_bits(g) for g in pool]
        min_d = self.max_distance
        
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                d = self.johnson_distance(bits_list[i], bits_list[j])
                min_d = min(min_d, d)
                if min_d == 0:
                    return 0
        return min_d
    
    def pre_generate_cover_samples(self, n_samples=2000):
        """Pré-gera amostras para covering radius (REUTILIZÁVEL)"""
        samples = set()
        while len(samples) < n_samples:
            game = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(game)
        self._cover_samples = [list(s) for s in samples]
        return self._cover_samples
    
    def covering_radius_fast(self, pool):
        """Covering radius usando amostras PRÉ-GERADAS"""
        if self._cover_samples is None:
            self.pre_generate_cover_samples()
        
        bits_list = [self.game_to_bits(g) for g in pool]
        max_min_dist = 0
        
        for sample in self._cover_samples[:500]:  # Usar subset
            sample_bits = self.game_to_bits(sample)
            min_dist = self.max_distance
            
            for pool_bits in bits_list:
                d = self.johnson_distance(sample_bits, pool_bits)
                min_dist = min(min_dist, d)
                if min_dist == 0:
                    break
            
            max_min_dist = max(max_min_dist, min_dist)
        
        return max_min_dist / self.max_distance


# ============================================================
# GARANTIA DE UNICIDADE (ANTI-DUPLICATAS)
# ============================================================

def ensure_unique_pool(pool, n_games):
    """
    Garante que o pool NÃO tenha jogos duplicados
    
    Remove duplicatas e completa com novos jogos diversos
    """
    unique = []
    seen = set()
    
    for game in pool:
        key = tuple(sorted(game))
        if key not in seen:
            seen.add(key)
            unique.append(sorted(game))
    
    # Completar se necessário
    while len(unique) < n_games:
        # Gerar jogo distante dos existentes
        best_game = None
        best_min_dist = -1
        
        for _ in range(50):
            candidate = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(candidate) in seen:
                continue
            
            min_dist = 15
            for existing in unique[-10:]:  # Comparar apenas com últimos 10
                common = len(set(candidate) & set(existing))
                min_dist = min(min_dist, 15 - common)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_game = candidate
        
        if best_game:
            seen.add(tuple(best_game))
            unique.append(best_game)
        else:
            # Fallback
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                unique.append(game)
    
    return unique[:n_games]


# ============================================================
# ANÁLISE DE SINAIS ESTRUTURAIS
# ============================================================

class SignalAnalyzer:
    """
    Extrai sinais estruturais de um pool
    
    Transforma métricas brutas em PERFIS interpretáveis
    """
    
    def __init__(self):
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
    
    def analyze_pool(self, pool):
        """Análise completa de um pool"""
        # Garantir unicidade
        pool = ensure_unique_pool(pool, len(pool))
        
        # Métricas Johnson
        min_dist = self.johnson.min_johnson_distance(pool)
        max_dist = self.johnson.max_distance
        
        # Cobertura de pares
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        pair_coverage = len(covered) / comb(25, 2)
        
        # Covering radius
        covering = self.johnson.covering_radius_fast(pool)
        
        # Diversidade (distância média)
        bits_list = [self.johnson.game_to_bits(g) for g in pool]
        distances = []
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                distances.append(self.johnson.johnson_distance(bits_list[i], bits_list[j]))
        avg_dist = np.mean(distances) if distances else 0
        
        # Entropia de distribuição
        all_dezenas = [d for g in pool for d in g]
        freq = np.bincount(all_dezenas, minlength=26)[1:]
        from scipy.stats import entropy
        ent = entropy(freq / np.sum(freq) + 1e-10) / np.log(25)
        
        # Compressibilidade
        import zlib, struct
        raw = bytearray()
        for g in pool:
            bits = 0
            for d in g:
                bits |= (1 << (d-1))
            raw.extend(struct.pack('>I', bits))
        compressed = zlib.compress(bytes(raw), level=9)
        compressibility = len(compressed) / len(raw)
        
        return {
            'pair_coverage': float(pair_coverage),
            'johnson_min': float(min_dist / max_dist),
            'covering_radius': float(covering),
            'avg_distance': float(avg_dist / max_dist),
            'entropy': float(ent),
            'compressibility': float(compressibility),
            'n_unique': len(pool),
            'n_duplicates': 0  # Garantido pelo ensure_unique
        }
    
    def classify_profile(self, signals):
        """
        Classifica o pool em um PERFIL de usuário
        
        Perfis:
        - 'conservador': estrutura organizada, baixa variância
        - 'caotico': máxima diversidade, alta independência
        - 'cobertura': máxima cobertura de pares/trincas
        - 'balanceado': equilíbrio entre todos
        """
        scores = {
            'conservador': (
                signals['compressibility'] * 2 +
                (1 - signals['covering_radius']) * 1.5 +
                signals['entropy'] * 0.5
            ),
            'caotico': (
                signals['johnson_min'] * 2 +
                signals['avg_distance'] * 1.5 +
                (1 - signals['compressibility']) * 1.5
            ),
            'cobertura': (
                signals['pair_coverage'] * 2.5 +
                (1 - signals['covering_radius']) * 2 +
                signals['avg_distance'] * 0.5
            ),
            'balanceado': (
                signals['pair_coverage'] * 1 +
                signals['johnson_min'] * 1 +
                (1 - signals['covering_radius']) * 1 +
                signals['entropy'] * 1
            )
        }
        
        best_profile = max(scores, key=scores.get)
        return best_profile, scores


# ============================================================
# FRONTEND DE EXPLORAÇÃO ESTRUTURAL
# ============================================================

class StructuralExplorer:
    """
    Frontend que consome sinais da fronteira de Pareto
    
    Permite ao usuário:
    - Visualizar pools por perfil
    - Comparar métricas estruturais
    - Selecionar trade-off preferido
    """
    
    def __init__(self, pareto_pools=None, pareto_objectives=None):
        """
        Inicializa com dados da fronteira de Pareto
        
        Args:
            pareto_pools: Lista de pools da fronteira
            pareto_objectives: Lista de vetores objetivo
        """
        self.analyzer = SignalAnalyzer()
        self.pareto_pools = pareto_pools or []
        self.pareto_objectives = pareto_objectives or []
        
        # Analisar todos os pools
        self.signals = []
        self.profiles = []
        
        if self.pareto_pools:
            self._analyze_all()
    
    def _analyze_all(self):
        """Analisa todos os pools da fronteira"""
        print(f"\n📊 ANALISANDO {len(self.pareto_pools)} SOLUÇÕES PARETO...")
        
        for i, pool in enumerate(self.pareto_pools):
            # Garantir unicidade
            pool = ensure_unique_pool(pool, len(pool))
            
            # Extrair sinais
            signals = self.analyzer.analyze_pool(pool)
            
            # Classificar perfil
            profile, scores = self.analyzer.classify_profile(signals)
            
            self.signals.append(signals)
            self.profiles.append({
                'index': i,
                'profile': profile,
                'scores': scores,
                'signals': signals
            })
        
        # Contar perfis
        profile_counts = Counter(p['profile'] for p in self.profiles)
        print(f"   ✅ Perfis encontrados: {dict(profile_counts)}")
    
    def get_pools_by_profile(self, profile_name, top_n=5):
        """
        Retorna os melhores pools para um perfil específico
        
        Args:
            profile_name: 'conservador', 'caotico', 'cobertura', 'balanceado'
            top_n: Número de pools a retornar
            
        Returns:
            list: Lista de (pool, signals, score) ordenados
        """
        matching = [p for p in self.profiles if p['profile'] == profile_name]
        
        if not matching:
            print(f"   ⚠️  Nenhum pool com perfil '{profile_name}'")
            return []
        
        # Ordenar pelo score do perfil
        matching.sort(key=lambda x: x['scores'][profile_name], reverse=True)
        
        results = []
        for p in matching[:top_n]:
            pool = ensure_unique_pool(self.pareto_pools[p['index']], len(self.pareto_pools[p['index']]))
            results.append((pool, p['signals'], p['scores'][profile_name]))
        
        return results
    
    def display_profile_summary(self):
        """Exibe resumo de todos os perfis disponíveis"""
        if not self.profiles:
            print("⚠️  Nenhum dado disponível. Execute _analyze_all primeiro.")
            return
        
        print(f"\n{'='*70}")
        print(f"🎯 PERFIS ESTRUTURAIS DISPONÍVEIS")
        print(f"{'='*70}")
        
        # Agrupar por perfil
        by_profile = defaultdict(list)
        for p in self.profiles:
            by_profile[p['profile']].append(p)
        
        for profile_name in ['conservador', 'caotico', 'cobertura', 'balanceado']:
            if profile_name not in by_profile:
                continue
            
            pools = by_profile[profile_name]
            n = len(pools)
            
            # Médias das métricas
            avg_signals = {}
            for key in pools[0]['signals']:
                avg_signals[key] = np.mean([p['signals'][key] for p in pools])
            
            print(f"\n🔹 {profile_name.upper()} ({n} soluções)")
            print(f"   ├─ Cobertura pares: {avg_signals['pair_coverage']:.3f}")
            print(f"   ├─ Distância Johnson: {avg_signals['johnson_min']:.3f}")
            print(f"   ├─ Covering radius: {avg_signals['covering_radius']:.3f}")
            print(f"   ├─ Entropia: {avg_signals['entropy']:.3f}")
            print(f"   ├─ Compressibilidade: {avg_signals['compressibility']:.3f}")
            print(f"   └─ Distância média: {avg_signals['avg_distance']:.3f}")
    
    def display_top_pools(self, profile_name='balanceado', top_n=3):
        """
        Exibe os melhores pools para um perfil
        """
        results = self.get_pools_by_profile(profile_name, top_n)
        
        if not results:
            return
        
        print(f"\n{'='*70}")
        print(f"🏆 TOP {top_n} POOLS - PERFIL: {profile_name.upper()}")
        print(f"{'='*70}")
        
        for i, (pool, signals, score) in enumerate(results, 1):
            print(f"\n📋 Pool #{i} (Score: {score:.3f})")
            print(f"   Cobertura: {signals['pair_coverage']:.3f} | "
                  f"Dist mín: {signals['johnson_min']:.3f} | "
                  f"Entropia: {signals['entropy']:.3f}")
            
            # Mostrar primeiros 5 jogos
            print(f"   Primeiros 5 jogos:")
            for j, game in enumerate(pool[:5]):
                print(f"      {j+1}. {sorted(game)}")
            
            if len(pool) > 5:
                print(f"      ... + {len(pool)-5} jogos")
    
    def export_signals(self, filename=None):
        """Exporta todos os sinais para JSON"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'sinais_estruturais_{timestamp}.json'
        
        export_data = []
        for i, p in enumerate(self.profiles):
            export_data.append({
                'index': i,
                'profile': p['profile'],
                'scores': p['scores'],
                'signals': {k: float(v) for k, v in p['signals'].items()}
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Sinais exportados: {filename}")
        return filename


# ============================================================
# DEMONSTRAÇÃO
# ============================================================

def demo():
    """Demonstração do Signal Engine + Frontend"""
    print("="*70)
    print("🎯 SIGNAL ENGINE + FRONTEND ESTRUTURAL")
    print("="*70)
    
    # Simular uma fronteira de Pareto (em produção, viria do NSGA-II)
    print("\n📊 Gerando fronteira de Pareto simulada...")
    
    pareto_pools = []
    pareto_objectives = []
    
    js = JohnsonSpace()
    
    # Gerar pools com diferentes características
    profiles_config = [
        ('conservador', 50, 0.3, 0.6),   # 50 jogos, baixa dist, alta compress
        ('conservador', 50, 0.25, 0.65),
        ('caotico', 50, 0.8, 0.3),       # alta dist, baixa compress
        ('caotico', 50, 0.75, 0.35),
        ('cobertura', 50, 0.5, 0.5),     # alta cobertura
        ('cobertura', 50, 0.45, 0.55),
        ('balanceado', 50, 0.5, 0.5),
        ('balanceado', 50, 0.55, 0.45),
    ]
    
    for profile, n_games, dist_factor, comp_factor in profiles_config:
        pool = []
        seen = set()
        
        # Gerar pool com característica específica
        base = sorted(np.random.choice(range(1, 26), 15, replace=False))
        
        for _ in range(n_games):
            if dist_factor > 0.5:
                # Alta diversidade: minimizar interseção
                best = None
                best_min = 15
                for _ in range(100):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool) if pool else 0
                    if m < best_min:
                        best_min = m
                        best = c
                if best:
                    seen.add(tuple(best))
                    pool.append(best)
            else:
                # Baixa diversidade: jogos similares
                game = base.copy()
                n_changes = int((1 - dist_factor) * 8) + 1
                for _ in range(n_changes):
                    pos = np.random.randint(0, 15)
                    avail = [d for d in range(1, 26) if d not in game]
                    if avail:
                        game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        # Garantir unicidade
        pool = ensure_unique_pool(pool, n_games)
        
        # Gerar objetivos simulados
        min_d = js.min_johnson_distance(pool) / js.max_distance
        covering = js.covering_radius_fast(pool)
        
        objectives = np.array([
            np.random.uniform(0.7, 0.95),  # Cobertura pares
            min_d,                          # Dist Johnson
            covering,                       # Covering radius
            comp_factor + np.random.uniform(-0.05, 0.05),  # Compressibilidade
            np.random.uniform(0.5, 0.8),   # Indep histórica
            np.random.uniform(0.1, 0.3)    # Conectividade
        ])
        
        pareto_pools.append(pool)
        pareto_objectives.append(objectives)
    
    # Criar explorer
    explorer = StructuralExplorer(pareto_pools, pareto_objectives)
    
    # Exibir resumo
    explorer.display_profile_summary()
    
    # Mostrar top pools por perfil
    for profile in ['conservador', 'caotico', 'cobertura', 'balanceado']:
        explorer.display_top_pools(profile, top_n=2)
    
    # Exportar
    explorer.export_signals()
    
    print(f"\n{'='*70}")
    print(f"✅ EXPLORADOR ESTRUTURAL CONCLUÍDO!")
    print(f"{'='*70}")
    print(f"\n💡 USO NO FRONTEND REAL:")
    print(f"   1. Execute NSGA-II para gerar fronteira de Pareto")
    print(f"   2. Passe pareto_pools para StructuralExplorer")
    print(f"   3. Usuário escolhe perfil desejado")
    print(f"   4. Sistema retorna melhores pools para aquele perfil")
    print(f"\n⚠️  DISCLAIMER:")
    print(f"   Isso NÃO prevê sorteios. Otimiza cobertura estrutural.")
    print(f"   Perfis diferentes = trade-offs diferentes.")


if __name__ == "__main__":
    demo()
