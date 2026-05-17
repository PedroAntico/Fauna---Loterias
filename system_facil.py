#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SISTEMA DE NAVEGAÇÃO NO ESPAÇO COMBINATÓRIO
============================================
Versão 8.0 - Embedding Estrutural + Scores Normalizados

MELHORIAS:
✅ Scores NORMALIZADOS (z-score por métrica)
✅ Entropia POSICIONAL (não global)
✅ UMAP embedding para visualização 2D/3D
✅ Clusters estruturais reais
✅ Perfis balanceados (anti-viés)
✅ Distância global (heapq, não últimos 10)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, zscore
from scipy.spatial.distance import pdist, squareform
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import warnings
import os
import json
import struct
import zlib
import heapq
from math import comb
from functools import lru_cache

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['figure.dpi'] = 120

# Tentar importar UMAP (opcional)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP não instalado. Use: pip install umap-learn")

# Tentar importar sklearn para PCA
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# JOHNSON SPACE (mantido)
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
        if len(pool) < 2: return self.max_distance
        bits_list = [self.game_to_bits(g) for g in pool]
        min_d = self.max_distance
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                d = self.johnson_distance(bits_list[i], bits_list[j])
                min_d = min(min_d, d)
                if min_d == 0: return 0
        return min_d
    
    def pre_generate_cover_samples(self, n_samples=2000):
        samples = set()
        while len(samples) < n_samples:
            game = tuple(sorted(np.random.choice(range(1, self.n+1), self.k, replace=False)))
            samples.add(game)
        self._cover_samples = [list(s) for s in samples]
        return self._cover_samples
    
    def covering_radius_fast(self, pool):
        if self._cover_samples is None:
            self.pre_generate_cover_samples()
        bits_list = [self.game_to_bits(g) for g in pool]
        max_min_dist = 0
        for sample in self._cover_samples[:500]:
            sample_bits = self.game_to_bits(sample)
            min_dist = self.max_distance
            for pool_bits in bits_list:
                d = self.johnson_distance(sample_bits, pool_bits)
                min_dist = min(min_dist, d)
                if min_dist == 0: break
            max_min_dist = max(max_min_dist, min_dist)
        return max_min_dist / self.max_distance


# ============================================================
# GARANTIA DE UNICIDADE (COM HEAPQ GLOBAL)
# ============================================================

def ensure_unique_pool(pool, n_games, johnson=None):
    """
    Garante unicidade com distância GLOBAL (heapq)
    Não apenas últimos 10 - considera geometria completa
    """
    if johnson is None:
        johnson = JohnsonSpace()
    
    unique = []
    seen = set()
    
    for game in pool:
        key = tuple(sorted(game))
        if key not in seen:
            seen.add(key)
            unique.append(sorted(game))
    
    while len(unique) < n_games:
        # Gerar candidatos
        candidates = []
        for _ in range(100):
            candidate = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(candidate) in seen:
                continue
            
            # Calcular distância MÍNIMA para TODOS os existentes
            bits_c = johnson.game_to_bits(candidate)
            min_dist = johnson.max_distance
            
            for existing in unique:
                bits_e = johnson.game_to_bits(existing)
                d = johnson.johnson_distance(bits_c, bits_e)
                min_dist = min(min_dist, d)
            
            # Usar heapq para manter os melhores
            if len(candidates) < 10:
                heapq.heappush(candidates, (-min_dist, tuple(candidate)))
            else:
                heapq.heappushpop(candidates, (-min_dist, tuple(candidate)))
        
        # Selecionar o mais distante
        if candidates:
            best = max(candidates, key=lambda x: -x[0])
            game = list(best[1])
            seen.add(tuple(game))
            unique.append(game)
        else:
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            if tuple(game) not in seen:
                seen.add(tuple(game))
                unique.append(game)
    
    return unique[:n_games]


# ============================================================
# ENTROPIA POSICIONAL (NÃO GLOBAL)
# ============================================================

def positional_entropy(pool):
    """
    Entropia por POSIÇÃO (mais discriminativa que global)
    
    Mede variabilidade em cada uma das 15 posições ordenadas
    """
    if not pool:
        return 0.0
    
    n_games = len(pool)
    pos_entropies = []
    
    # Para cada posição (1ª a 15ª dezena ordenada)
    for pos in range(15):
        # Extrair dezenas nesta posição
        pos_values = [sorted(g)[pos] for g in pool]
        
        # Frequência
        freq = np.bincount(pos_values, minlength=26)[1:]
        probs = freq / n_games
        probs = np.where(probs > 0, probs, 1e-10)
        
        # Entropia da posição
        pos_ent = entropy(probs)
        pos_entropies.append(pos_ent)
    
    # Média das entropias posicionais (normalizada)
    max_entropy = np.log(25)
    avg_pos_entropy = np.mean(pos_entropies) / max_entropy
    
    return float(avg_pos_entropy)


def mutual_information_positions(pool):
    """
    Informação mútua entre posições adjacentes
    
    Mede dependência estrutural
    """
    if len(pool) < 2:
        return 0.0
    
    mi_values = []
    
    for pos in range(14):  # Pares de posições adjacentes
        pos1_vals = [sorted(g)[pos] for g in pool]
        pos2_vals = [sorted(g)[pos+1] for g in pool]
        
        # Tabela de contingência
        contingency = np.zeros((25, 25))
        for v1, v2 in zip(pos1_vals, pos2_vals):
            contingency[v1-1, v2-1] += 1
        
        # Informação mútua
        joint = contingency / len(pool)
        marginal1 = joint.sum(axis=1)
        marginal2 = joint.sum(axis=0)
        
        mi = 0.0
        for i in range(25):
            for j in range(25):
                if joint[i, j] > 0:
                    expected = marginal1[i] * marginal2[j]
                    if expected > 0:
                        mi += joint[i, j] * np.log(joint[i, j] / expected)
        
        mi_values.append(mi)
    
    return float(np.mean(mi_values))


# ============================================================
# ANALISADOR DE SINAIS (COM SCORES NORMALIZADOS)
# ============================================================

class SignalAnalyzer:
    """
    Extrai sinais estruturais com métricas BALANCEADAS
    
    Correções:
    - Entropia posicional (não global)
    - Scores normalizados (z-score)
    - Pesos calibrados para não enviesar
    """
    
    def __init__(self):
        self.johnson = JohnsonSpace()
        self.johnson.pre_generate_cover_samples(2000)
    
    def analyze_pool(self, pool):
        """Análise completa com métricas discriminativas"""
        pool = ensure_unique_pool(pool, len(pool), self.johnson)
        
        # Cobertura de pares
        covered = set()
        for game in pool:
            for pair in combinations(sorted(game), 2):
                covered.add(pair)
        pair_coverage = len(covered) / comb(25, 2)
        
        # Distância Johnson
        min_dist = self.johnson.min_johnson_distance(pool)
        max_dist = self.johnson.max_distance
        
        # Covering radius
        covering = self.johnson.covering_radius_fast(pool)
        
        # Distância média
        bits_list = [self.johnson.game_to_bits(g) for g in pool]
        distances = []
        for i in range(len(bits_list)):
            for j in range(i+1, len(bits_list)):
                distances.append(self.johnson.johnson_distance(bits_list[i], bits_list[j]))
        avg_dist = np.mean(distances) if distances else 0
        
        # ENTROPIA POSICIONAL (não global)
        pos_entropy = positional_entropy(pool)
        
        # Informação mútua (dependência estrutural)
        mi = mutual_information_positions(pool)
        
        # Compressibilidade
        raw = bytearray()
        for g in pool:
            bits = 0
            for d in g:
                bits |= (1 << (d-1))
            raw.extend(struct.pack('>I', bits))
        compressed = zlib.compress(bytes(raw), level=9)
        compressibility = len(compressed) / len(raw)
        
        # Variância da distância (nova métrica)
        std_dist = np.std(distances) if distances else 0
        
        return {
            'pair_coverage': float(pair_coverage),
            'johnson_min': float(min_dist / max_dist),
            'covering_radius': float(covering),
            'avg_distance': float(avg_dist / max_dist),
            'std_distance': float(std_dist / max_dist),
            'pos_entropy': float(pos_entropy),
            'mutual_info': float(mi),
            'compressibility': float(compressibility),
            'n_unique': len(pool)
        }
    
    def classify_profile(self, signals, all_signals_list=None):
        """
        Classificação BALANCEADA com normalização
        
        Se all_signals_list fornecida, usa z-score para normalizar
        """
        scores = {}
        
        # Se temos lista de todos os sinais, normalizar
        if all_signals_list and len(all_signals_list) > 1:
            normalized = self._normalize_signals(signals, all_signals_list)
        else:
            normalized = signals
        
        # Pesos CALIBRADOS (todos ~mesma magnitude)
        scores['conservador'] = (
            (1 - normalized['pos_entropy']) * 1.5 +
            normalized['compressibility'] * 1.5 +
            (1 - normalized['std_distance']) * 1.0
        )
        
        scores['caotico'] = (
            normalized['johnson_min'] * 1.5 +
            normalized['avg_distance'] * 1.5 +
            normalized['pos_entropy'] * 1.0
        )
        
        scores['cobertura'] = (
            normalized['pair_coverage'] * 1.5 +
            (1 - normalized['covering_radius']) * 1.5 +
            normalized['avg_distance'] * 1.0
        )
        
        scores['balanceado'] = (
            normalized['pair_coverage'] * 1.0 +
            normalized['johnson_min'] * 1.0 +
            (1 - normalized['covering_radius']) * 1.0 +
            normalized['pos_entropy'] * 1.0
        )
        
        best_profile = max(scores, key=scores.get)
        return best_profile, scores
    
    def _normalize_signals(self, signals, all_signals):
        """
        Normaliza sinais usando estatísticas da população
        
        Converte para escala 0-1 relativa
        """
        normalized = {}
        
        for key in signals:
            values = [s[key] for s in all_signals]
            vmin = np.min(values)
            vmax = np.max(values)
            
            if vmax - vmin > 1e-10:
                normalized[key] = (signals[key] - vmin) / (vmax - vmin)
            else:
                normalized[key] = 0.5
        
        return normalized


# ============================================================
# EMBEDDING ESTRUTURAL (UMAP/PCA)
# ============================================================

class StructuralEmbedding:
    """
    Embedding 2D/3D do espaço de soluções
    
    Revela clusters, outliers, e geometria dos perfis
    """
    
    def __init__(self, signals_list):
        """
        Args:
            signals_list: Lista de dicionários de sinais
        """
        self.signals_list = signals_list
        self.feature_names = [
            'pair_coverage', 'johnson_min', 'covering_radius',
            'avg_distance', 'std_distance', 'pos_entropy',
            'mutual_info', 'compressibility'
        ]
        
        # Matriz de features
        self.X = self._build_feature_matrix()
        
        # Embeddings
        self.embedding_2d = None
        self.embedding_3d = None
        
    def _build_feature_matrix(self):
        """Constrói matriz de features"""
        X = np.zeros((len(self.signals_list), len(self.feature_names)))
        
        for i, signals in enumerate(self.signals_list):
            for j, name in enumerate(self.feature_names):
                X[i, j] = signals.get(name, 0.0)
        
        return X
    
    def compute_umap(self, n_components=2, random_state=42):
        """Computa UMAP embedding"""
        if not UMAP_AVAILABLE:
            print("⚠️  UMAP não disponível. Usando PCA.")
            return self.compute_pca(n_components)
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=min(15, len(self.X) - 1),
            min_dist=0.1,
            metric='euclidean'
        )
        
        embedding = reducer.fit_transform(X_scaled)
        
        if n_components == 2:
            self.embedding_2d = embedding
        else:
            self.embedding_3d = embedding
        
        return embedding
    
    def compute_pca(self, n_components=2):
        """Computa PCA embedding (fallback)"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(X_scaled)
        
        if n_components == 2:
            self.embedding_2d = embedding
        else:
            self.embedding_3d = embedding
        
        # Variância explicada
        print(f"   PCA: {pca.explained_variance_ratio_.sum()*100:.1f}% variância")
        
        return embedding
    
    def visualize(self, profiles=None, output_dir='graficos_embedding'):
        """Visualiza embedding 2D com perfis"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.embedding_2d is None:
            self.compute_umap(2)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # 1. Embedding colorido por perfil
        ax = axes[0]
        
        if profiles:
            profile_colors = {
                'conservador': 'blue',
                'caotico': 'red',
                'cobertura': 'green',
                'balanceado': 'purple'
            }
            
            for profile_name, color in profile_colors.items():
                mask = [p == profile_name for p in profiles]
                if any(mask):
                    ax.scatter(
                        self.embedding_2d[mask, 0],
                        self.embedding_2d[mask, 1],
                        c=color, label=profile_name,
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5
                    )
            
            ax.legend()
        else:
            ax.scatter(
                self.embedding_2d[:, 0],
                self.embedding_2d[:, 1],
                c='blue', s=100, alpha=0.7
            )
        
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('Embedding Estrutural - Espaço de Soluções')
        ax.grid(True, alpha=0.3)
        
        # 2. Heatmap das features
        ax = axes[1]
        
        # Ordenar por perfil para visualização
        if profiles:
            profile_order = {'conservador': 0, 'balanceado': 1, 'cobertura': 2, 'caotico': 3}
            order = sorted(range(len(profiles)), key=lambda i: profile_order.get(profiles[i], 99))
            X_ordered = self.X[order]
        else:
            X_ordered = self.X
        
        im = ax.imshow(X_ordered.T, aspect='auto', cmap='RdYlGn')
        ax.set_xlabel('Solução')
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels([n[:15] for n in self.feature_names])
        ax.set_title('Matriz de Sinais Estruturais')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle('Navegação no Espaço Combinatório', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/embedding_estrutural.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        print(f"✅ Gráficos salvos em {output_dir}/")
        return fig


# ============================================================
# FRONTEND DE NAVEGAÇÃO
# ============================================================

class StructuralNavigator:
    """
    Navegador interativo no espaço combinatório
    
    Usa embedding + perfis para exploração
    """
    
    def __init__(self, pareto_pools=None):
        self.analyzer = SignalAnalyzer()
        self.pareto_pools = pareto_pools or []
        
        # Analisar todos
        self.all_signals = []
        self.all_profiles = []
        
        if self.pareto_pools:
            self._analyze_all()
    
    def _analyze_all(self):
        """Analisa todos os pools com normalização populacional"""
        print(f"\n📊 ANALISANDO {len(self.pareto_pools)} SOLUÇÕES...")
        
        # Primeiro passo: extrair sinais brutos
        raw_signals = []
        for pool in self.pareto_pools:
            signals = self.analyzer.analyze_pool(pool)
            raw_signals.append(signals)
        
        # Segundo passo: classificar COM normalização
        for i, signals in enumerate(raw_signals):
            profile, scores = self.analyzer.classify_profile(signals, raw_signals)
            
            self.all_signals.append(signals)
            self.all_profiles.append({
                'index': i,
                'profile': profile,
                'scores': scores,
                'signals': signals
            })
        
        # Estatísticas
        profile_counts = Counter(p['profile'] for p in self.all_profiles)
        print(f"   ✅ Perfis: {dict(profile_counts)}")
    
    def get_pools_by_profile(self, profile_name, top_n=5):
        """Retorna melhores pools para um perfil"""
        matching = [p for p in self.all_profiles if p['profile'] == profile_name]
        
        if not matching:
            return []
        
        matching.sort(key=lambda x: x['scores'][profile_name], reverse=True)
        
        results = []
        for p in matching[:top_n]:
            pool = self.pareto_pools[p['index']]
            results.append((pool, p['signals'], p['scores'][profile_name]))
        
        return results
    
    def create_embedding(self):
        """Cria embedding estrutural"""
        embedding = StructuralEmbedding(self.all_signals)
        profiles = [p['profile'] for p in self.all_profiles]
        embedding.compute_umap(2)
        embedding.visualize(profiles)
        return embedding
    
    def display_dashboard(self):
        """Dashboard completo"""
        print(f"\n{'='*70}")
        print(f"🎯 DASHBOARD DE NAVEGAÇÃO ESTRUTURAL")
        print(f"{'='*70}")
        print(f"\n📊 {len(self.pareto_pools)} soluções analisadas")
        
        # Distribuição de perfis
        profile_counts = Counter(p['profile'] for p in self.all_profiles)
        print(f"\n📊 DISTRIBUIÇÃO DE PERFIS:")
        for profile, count in profile_counts.most_common():
            bar = '█' * (count * 40 // len(self.all_profiles))
            print(f"   {profile:<15} {bar} {count}")
        
        # Estatísticas por perfil
        print(f"\n📊 MÉTRICAS POR PERFIL:")
        
        by_profile = defaultdict(list)
        for p in self.all_profiles:
            by_profile[p['profile']].append(p)
        
        for profile_name in ['conservador', 'caotico', 'cobertura', 'balanceado']:
            if profile_name not in by_profile:
                continue
            
            pools = by_profile[profile_name]
            n = len(pools)
            
            avg = {}
            for key in pools[0]['signals']:
                avg[key] = np.mean([p['signals'][key] for p in pools])
            
            print(f"\n   {profile_name.upper()} ({n} pools):")
            print(f"      Cobertura: {avg['pair_coverage']:.3f}")
            print(f"      Dist mín:  {avg['johnson_min']:.3f}")
            print(f"      Cov radius:{avg['covering_radius']:.3f}")
            print(f"      Entropia:  {avg['pos_entropy']:.3f}")
            print(f"      MI:        {avg['mutual_info']:.4f}")
    
    def display_top_pools(self, profile_name='balanceado', top_n=3):
        """Exibe top pools para perfil"""
        results = self.get_pools_by_profile(profile_name, top_n)
        
        if not results:
            print(f"   ⚠️  Nenhum pool com perfil '{profile_name}'")
            return
        
        print(f"\n{'='*70}")
        print(f"🏆 TOP {top_n} - {profile_name.upper()}")
        print(f"{'='*70}")
        
        for i, (pool, signals, score) in enumerate(results, 1):
            print(f"\n📋 #{i} (Score: {score:.3f})")
            print(f"   Cob: {signals['pair_coverage']:.3f} | "
                  f"Dist: {signals['johnson_min']:.3f} | "
                  f"Ent: {signals['pos_entropy']:.3f} | "
                  f"MI: {signals['mutual_info']:.4f}")
            
            for j, game in enumerate(pool[:5]):
                print(f"      {j+1}. {sorted(game)}")
            if len(pool) > 5:
                print(f"      ... +{len(pool)-5}")


def demo():
    """Demonstração completa"""
    print("="*70)
    print("🧭 NAVEGADOR NO ESPAÇO COMBINATÓRIO")
    print("="*70)
    
    # Gerar pools simulados com diversidade REAL
    print("\n📊 Gerando soluções diversas...")
    
    js = JohnsonSpace()
    pareto_pools = []
    
    # Estratégias DIFERENTES
    strategies = [
        ('spread', 50),      # Máxima diversidade
        ('spread', 50),
        ('clustered', 50),   # Baixa diversidade
        ('clustered', 50),
        ('balanced', 50),    # Balanceado
        ('balanced', 50),
        ('random', 50),      # Aleatório
        ('random', 50),
    ]
    
    for strategy, n_games in strategies:
        pool = []
        seen = set()
        
        if strategy == 'spread':
            # Maximizar distância
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            seen.add(tuple(base))
            
            for _ in range(n_games - 1):
                best = None
                best_min = 15
                for _ in range(200):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                if best:
                    seen.add(tuple(best))
                    pool.append(best)
        
        elif strategy == 'clustered':
            # Jogos similares
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(n_games):
                game = base.copy()
                n_swap = np.random.randint(1, 4)
                for _ in range(n_swap):
                    pos = np.random.randint(0, 15)
                    avail = [d for d in range(1, 26) if d not in game]
                    if avail:
                        game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        elif strategy == 'balanced':
            # Metade spread, metade clustered
            half = n_games // 2
            base = sorted(np.random.choice(range(1, 26), 15, replace=False))
            pool.append(base)
            seen.add(tuple(base))
            
            for _ in range(half - 1):
                best = None
                best_min = 15
                for _ in range(100):
                    c = sorted(np.random.choice(range(1, 26), 15, replace=False))
                    if tuple(c) in seen: continue
                    m = min(len(set(c) & set(g)) for g in pool)
                    if m < best_min:
                        best_min = m
                        best = c
                if best:
                    seen.add(tuple(best))
                    pool.append(best)
            
            cbase = sorted(np.random.choice(range(1, 26), 15, replace=False))
            for _ in range(n_games - half):
                game = cbase.copy()
                pos = np.random.randint(0, 15)
                avail = [d for d in range(1, 26) if d not in game]
                if avail:
                    game[pos] = np.random.choice(avail)
                game = sorted(game)
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        else:  # random
            for _ in range(n_games):
                game = sorted(np.random.choice(range(1, 26), 15, replace=False))
                if tuple(game) not in seen:
                    seen.add(tuple(game))
                    pool.append(game)
        
        pool = ensure_unique_pool(pool, n_games, js)
        pareto_pools.append(pool)
    
    # Criar navegador
    nav = StructuralNavigator(pareto_pools)
    
    # Dashboard
    nav.display_dashboard()
    
    # Top pools por perfil
    for profile in ['conservador', 'caotico', 'cobertura', 'balanceado']:
        nav.display_top_pools(profile, top_n=2)
    
    # Embedding
    print(f"\n📐 CRIANDO EMBEDDING ESTRUTURAL...")
    try:
        nav.create_embedding()
    except Exception as e:
        print(f"   ⚠️  Embedding falhou: {e}")
        print(f"   💡 Isso é apenas visualização, não afeta o motor.")
    
    print(f"\n{'='*70}")
    print(f"✅ NAVEGAÇÃO CONCLUÍDA!")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
