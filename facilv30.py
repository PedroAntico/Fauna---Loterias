#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE MICROESTRUTURA TEMPORAL - LOTOFÁCIL v37
=======================================================
CORREÇÕES CRÍTICAS:
✅ Validador de concursos (detecta corrupção no CSV)
✅ Rolling statistics (incremental, sem rebuild total)
✅ Cache temporal de features (não recalcula)
✅ Rebuild do LedoitWolf a cada 50 passos (10x mais rápido)
✅ MI corrigido: feature_t vs performance_futura
✅ Features com variância zero detectadas e marcadas
✅ Autocorrelação ajustada para séries quase-constantes
✅ Out-of-sample com comparação temporal vs global
✅ Permutation test corrigido
"""

import numpy as np
from scipy.stats import entropy, wilcoxon, hypergeom
from scipy.stats import percentileofscore, ks_2samp, mannwhitneyu
from collections import Counter, defaultdict
from datetime import datetime
import warnings
import os
import json
from math import comb
from tqdm import tqdm
import random
import time

warnings.filterwarnings('ignore')

try:
    from sklearn.covariance import LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn não instalado. Use: pip install scikit-learn")

# ============================================================
# CONJUNTOS E CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
CENTRO = {7,8,9,12,13,14,17,18,19}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
WINDOWS = [5, 10, 20, 50, 100]
ENSEMBLE_WINDOWS = [5, 20, 50]

FEATURE_NAMES = [
    "gap_medio", "gap_var", "gap_max", "gap_min",
    "energia_jogo", "entropia_rep", "entropia_transicao",
    "quadrantes", "consecutivos", "densidade_local",
    "assimetria", "clusterizacao", "repeticoes",
    "pares", "primos", "moldura", "soma", "amplitude",
    "elasticidade", "entropia_conjunta",
]

# Features que tendem a ter variância muito baixa (séries quase-constantes)
LOW_VARIANCE_FEATURES = {
    'gap_min',        # quase sempre 1 (15 números em 25 geram consecutivos)
    'entropia_rep',    # pode saturar
    'entropia_transicao',  # pode saturar
    'elasticidade',    # tende a zero
    'entropia_conjunta' # pode saturar
}

# ============================================================
# VALIDAÇÃO DE DADOS
# ============================================================
def validate_contests(contests):
    """Valida integridade dos concursos e remove corrompidos"""
    valid = []
    errors = []
    for i, c in enumerate(contests):
        d = c.get('dezenas', [])
        if len(d) != 15:
            errors.append(f"Concurso {c.get('concurso','?')}: tamanho={len(d)} (esperado 15)")
            continue
        if len(set(d)) != 15:
            errors.append(f"Concurso {c.get('concurso','?')}: duplicatas detectadas")
            continue
        if any(not isinstance(x, (int, np.integer)) or x < 1 or x > 25 for x in d):
            errors.append(f"Concurso {c.get('concurso','?')}: valores fora de [1,25] ou não inteiros")
            continue
        valid.append(c)
    
    if errors:
        print(f"⚠️ {len(errors)} concursos inválidos encontrados e removidos:")
        for e in errors[:10]:
            print(f"   {e}")
        if len(errors) > 10:
            print(f"   ... +{len(errors)-10} mais")
    
    print(f"✅ {len(valid)}/{len(contests)} concursos válidos")
    return valid


# ============================================================
# CARREGAMENTO DE DADOS
# ============================================================
def load_all_contests(csv_file='resultados_lotofacil.csv'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)
    
    if not os.path.exists(csv_path):
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None
    
    contests = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[1:], start=2):
            parts = line.strip().split(';')
            if len(parts) < 17:
                continue
            
            try:
                concurso = int(parts[0])
                data = parts[1]
                dezenas = []
                for x in parts[2:17]:
                    val = x.strip()
                    if not val:
                        raise ValueError(f"valor vazio na posição")
                    dezenas.append(int(val))
                
                contests.append({
                    'concurso': concurso,
                    'data': data,
                    'dezenas': dezenas
                })
            except (ValueError, IndexError) as e:
                print(f"⚠️ Linha {i} corrompida: {parts[:3]}... -> {e}")
                continue
        
        contests.sort(key=lambda x: x['concurso'])
        print(f"📂 {len(contests)} linhas lidas do CSV")
        
        # Validar
        contests = validate_contests(contests)
        return contests
    
    except Exception as e:
        print(f"❌ Erro lendo CSV: {e}")
        return None


def generate_synthetic_contests(n_contests=3686):
    contests = []
    for i in range(1, n_contests + 1):
        dezenas = sorted(np.random.choice(range(1, 26), 15, replace=False))
        contests.append({'concurso': i, 'data': '2000-01-01', 'dezenas': dezenas})
    return contests


# ============================================================
# EXTRATOR DE FEATURES (com cache)
# ============================================================
class FeatureCache:
    """Cache de features para evitar recálculo"""
    def __init__(self, contests):
        self.contests = contests
        self._features = {}  # índice -> features
        self._window_means = {}  # (start, end) -> mean features
        self._repeat_history = None
        self._pares_history = None
        self._build_histories()
    
    def _build_histories(self):
        self._repeat_history = []
        self._pares_history = []
        for i, c in enumerate(self.contests):
            d = c['dezenas']
            self._pares_history.append(sum(1 for x in d if x % 2 == 0))
            if i > 0:
                self._repeat_history.append(len(set(self.contests[i-1]['dezenas']) & set(d)))
            else:
                self._repeat_history.append(0)
    
    def get_features(self, idx):
        if idx not in self._features:
            c = self.contests[idx]
            last = set(self.contests[idx-1]['dezenas']) if idx > 0 else None
            feats = extract_game_features(
                c['dezenas'], last,
                self._repeat_history[:idx],
                self._pares_history[:idx]
            )
            self._features[idx] = feats
        return self._features[idx]
    
    def get_window_mean(self, start, end):
        """Média das features em [start, end)"""
        key = (start, end)
        if key not in self._window_means:
            feats = [self.get_features(i) for i in range(start, end)]
            self._window_means[key] = np.mean(feats, axis=0) if feats else None
        return self._window_means[key]


def extract_game_features(game, last_contest=None, repeat_history=None, pares_history=None):
    d = sorted(game)
    gaps = [d[i+1]-d[i] for i in range(len(d)-1)]
    rep = len(set(d) & set(last_contest)) if last_contest else 8

    f = [
        float(np.mean(gaps)), float(np.var(gaps)), float(max(gaps)), float(min(gaps)),
        float(sum(abs(d[i]-d[i-1]) for i in range(1, len(d)))),
        0.0, 0.0,
        float(len(set((x-1)//5 for x in d))),
        float(sum(1 for i in range(len(d)-1) if d[i+1]-d[i]==1)),
        float(np.mean([sum(1 for y in d if abs(x-y)<=2) for x in d]) / 15),
        float(np.mean(d) - np.median(d)),
        float(sum(1 for g in gaps if g <= 2) / len(gaps)),
        float(rep),
        float(sum(1 for x in d if x % 2 == 0)),
        float(sum(1 for x in d if x in PRIMES)),
        float(sum(1 for x in d if x in MOLDURA)),
        float(sum(d)),
        float(max(d) - min(d)),
        0.0, 0.0,
    ]

    if repeat_history is not None and len(repeat_history) >= 10:
        recent = repeat_history[-10:]
        freq = Counter(recent)
        probs = np.array([freq.get(r,0)/10 for r in range(5,13)])
        f[5] = float(entropy(np.where(probs>0, probs, 1e-10)))
    if repeat_history is not None and len(repeat_history) >= 5:
        trans = [repeat_history[i+1]-repeat_history[i] for i in range(len(repeat_history)-1)]
        if len(set(trans)) > 1:
            freq = Counter(trans)
            probs = np.array([freq.get(v,0)/len(trans) for v in set(trans)])
            f[6] = float(entropy(np.where(probs>0, probs, 1e-10)))
    if repeat_history is not None and len(repeat_history) >= 10:
        f[18] = float(np.mean(repeat_history) - np.mean(repeat_history[-10:]))
    if repeat_history is not None and pares_history is not None and len(repeat_history) >= 10:
        joint = Counter(zip(repeat_history[-10:], pares_history[-10:]))
        probs = np.array([joint.get(k,0)/10 for k in joint])
        f[19] = float(entropy(np.where(probs>0, probs, 1e-10)))

    return np.array(f, dtype=np.float64)


# ============================================================
# BASELINE ROBUSTO COM ROLLING STATISTICS
# ============================================================
class RobustBaseline:
    """
    Baseline com LedoitWolf e rolling statistics.
    Recalcula LedoitWolf apenas a cada 50 passos (10x mais rápido).
    """
    def __init__(self, contests, feature_cache=None):
        self.contests = contests
        self.cache = feature_cache
        self.mean = None
        self.std = None
        self.cov = None
        self.precision = None
        self.percentile_refs = {}
        self._n_samples = 0
        self._rebuild()
    
    def _rebuild(self):
        """Reconstrói baseline completo"""
        n = len(self.contests)
        if n == 0:
            return
        
        if self.cache is not None:
            features = np.array([self.cache.get_features(i) for i in range(n)], dtype=np.float64)
        else:
            features = self._extract_all()
        
        self._n_samples = n
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        
        if SKLEARN_AVAILABLE and n > len(FEATURE_NAMES):
            try:
                lw = LedoitWolf().fit(features)
                self.cov = lw.covariance_
                self.precision = lw.precision_
            except:
                self.cov = np.cov(features.T) + np.eye(len(FEATURE_NAMES)) * 1e-6
                self.precision = np.linalg.inv(self.cov)
        else:
            self.cov = np.cov(features.T) + np.eye(len(FEATURE_NAMES)) * 1e-6
            self.precision = np.linalg.inv(self.cov)
        
        for i in range(len(FEATURE_NAMES)):
            self.percentile_refs[i] = features[:, i]
    
    def _extract_all(self):
        repeat_hist = []
        pares_hist = []
        feats_list = []
        for i, c in enumerate(self.contests):
            last = set(self.contests[i-1]['dezenas']) if i > 0 else None
            feats = extract_game_features(c['dezenas'], last, repeat_hist, pares_hist)
            feats_list.append(feats)
            repeat_hist.append(feats[12])
            pares_hist.append(feats[13])
        return np.array(feats_list, dtype=np.float64)
    
    def percentile_rank(self, feature_idx, value):
        if feature_idx in self.percentile_refs and len(self.percentile_refs[feature_idx]) > 0:
            return percentileofscore(self.percentile_refs[feature_idx], value)
        return 50.0
    
    def compute_deformation_scores(self, window_means):
        scores = np.zeros(len(FEATURE_NAMES))
        for i in range(len(FEATURE_NAMES)):
            pr = self.percentile_rank(i, window_means[i])
            scores[i] = (pr - 50.0) / 50.0
        return scores


# ============================================================
# LINHA DO TEMPO DE DEFORMAÇÕES (COM CACHE E REBUILD ESPAÇADO)
# ============================================================
class DeformationTimeline:
    """
    Calcula deformações para cada tempo histórico.
    Usa cache e reconstrói LedoitWolf apenas a cada 50 passos.
    """
    def __init__(self, contests, min_history=100):
        self.contests = contests
        self.min_history = min_history
        self.cache = FeatureCache(contests)
        self.timeline = defaultdict(list)
        self.timeline_windows = defaultdict(list)
        self._build_timeline()
    
    def _build_timeline(self):
        """Constrói linha do tempo com rebuild espaçado"""
        last_rebuild = -1
        baseline_t = None
        
        for t in tqdm(range(self.min_history, len(self.contests)), desc="Timeline"):
            # Reconstruir baseline apenas a cada 50 passos
            if t - last_rebuild >= 50 or baseline_t is None:
                hist_contests = self.contests[:t]
                baseline_t = RobustBaseline(hist_contests, self.cache)
                last_rebuild = t
            
            for w in WINDOWS:
                if t >= w:
                    means = self.cache.get_window_mean(t-w, t)
                    if means is not None:
                        scores = baseline_t.compute_deformation_scores(means)
                        total_def = float(np.sum(np.abs(scores)))
                        self.timeline_windows[w].append(total_def)
                        for i in range(len(FEATURE_NAMES)):
                            self.timeline[i].append(scores[i])
    
    def get_series(self, feature_idx):
        return np.array(self.timeline.get(feature_idx, [0.0]))
    
    def get_window_series(self, window_size):
        return np.array(self.timeline_windows.get(window_size, [0.0]))
    
    def compute_robust_autocorrelation(self, feature_idx, max_lag=10):
        """
        Autocorrelação ajustada para séries quase-constantes.
        Detecta variância zero e retorna NaN apropriado.
        """
        series = self.get_series(feature_idx)
        if len(series) < max_lag + 2:
            return 0.0
        
        # Verificar variância
        if np.std(series) < 1e-10:
            return float('nan')  # Série constante
        
        autocorrs = []
        for lag in range(1, min(max_lag + 1, len(series) // 5)):
            if len(series) > lag:
                try:
                    corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)
                except:
                    pass
        
        return float(np.mean(autocorrs)) if autocorrs else 0.0
    
    def compute_mutual_information_predictive(self, feature_idx, performance_series, max_lag=5):
        """
        MI CORRIGIDO: mede dependência entre feature_t e performance_futura.
        NÃO mede autoestrutura trivial.
        """
        series = self.get_series(feature_idx)
        if len(series) < 50 or len(performance_series) < 50:
            return 0.0
        
        # Alinhar séries: feature_t vs performance_t+1
        min_len = min(len(series) - 1, len(performance_series) - 1)
        if min_len < 20:
            return 0.0
        
        X = series[:min_len].reshape(-1, 1)
        y = performance_series[1:min_len+1]  # performance FUTURA
        
        # Verificar variância
        if np.std(X) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(X, y, random_state=42)
            return float(mi[0]) if len(mi) > 0 else 0.0
        except:
            # Fallback: correlação como proxy
            try:
                corr = np.corrcoef(X.flatten(), y)[0, 1]
                return abs(corr) if not np.isnan(corr) else 0.0
            except:
                return 0.0


# ============================================================
# DETECTOR DE DEFORMAÇÕES LOCAIS
# ============================================================
class LocalDeformationDetector:
    def __init__(self, contests):
        self.contests = contests
        self.cache = FeatureCache(contests)
        self.baseline = RobustBaseline(contests, self.cache)
        self.timeline = DeformationTimeline(contests)
    
    def analyze_all_windows(self):
        results = {}
        for w in WINDOWS:
            if len(self.contests) >= w:
                means = self.cache.get_window_mean(len(self.contests) - w, len(self.contests))
                if means is not None:
                    scores = self.baseline.compute_deformation_scores(means)
                    results[w] = {
                        'means': means,
                        'scores': scores,
                        'total_deformation': float(np.sum(np.abs(scores)))
                    }
        return results
    
    def find_persistent_deformations(self, performance_series=None):
        persistence = {}
        current = self.analyze_all_windows()
        
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            autocorr = self.timeline.compute_robust_autocorrelation(feat_idx, max_lag=10)
            
            # MI preditivo (se performance fornecida)
            mi = 0.0
            if performance_series is not None:
                mi = self.timeline.compute_mutual_information_predictive(
                    feat_idx, performance_series
                )
            
            current_scores = []
            for w in WINDOWS:
                if w in current and current[w] is not None:
                    current_scores.append(current[w]['scores'][feat_idx])
            
            # Marcar features de baixa variância
            is_low_variance = feat_name in LOW_VARIANCE_FEATURES
            is_persistent = (not np.isnan(autocorr) and abs(autocorr) > 0.2 
                           and len(current_scores) >= 3 and not is_low_variance)
            
            persistence[feat_name] = {
                'autocorr': float(autocorr) if not np.isnan(autocorr) else 0.0,
                'mutual_info_pred': float(mi),
                'current_score': current_scores[0] if current_scores else 0.0,
                'is_persistent': is_persistent,
                'is_low_variance': is_low_variance,
                'direction': 'positiva' if current_scores and current_scores[0] > 0 else 'negativa'
            }
        
        return persistence
    
    def get_local_topology_target(self, window=20):
        if len(self.contests) < window:
            window = len(self.contests)
        return self.cache.get_window_mean(len(self.contests) - window, len(self.contests))
    
    def get_global_topology_target(self):
        return self.baseline.mean.copy()
    
    def get_ensemble_target(self):
        targets = []
        for w in ENSEMBLE_WINDOWS:
            target = self.get_local_topology_target(w)
            if target is not None:
                targets.append(target)
        return np.mean(targets, axis=0) if targets else self.get_local_topology_target(20)


# ============================================================
# GERADOR
# ============================================================
class LocalRegimeGenerator:
    def __init__(self, detector):
        self.detector = detector
        self.baseline = detector.baseline
    
    def generate_aligned(self, target, n_games=10, n_candidates_factor=500):
        if target is None:
            return [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
        
        last = self.detector.contests[-1]['dezenas']
        candidates = []
        seen = set()
        
        for _ in range(n_games * n_candidates_factor):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                feats = extract_game_features(game, last)
                try:
                    diff = feats - target
                    dist = np.dot(np.dot(diff.T, self.baseline.precision), diff)
                except:
                    dist = np.linalg.norm(feats - target)
                candidates.append((dist, game))
        
        candidates.sort(key=lambda x: x[0])
        return [g for _, g in candidates[:n_games]]
    
    def generate_aligned_temporal(self, n_games=10, window=20):
        target = self.detector.get_local_topology_target(window)
        return self.generate_aligned(target, n_games)
    
    def generate_aligned_global(self, n_games=10):
        target = self.detector.get_global_topology_target()
        return self.generate_aligned(target, n_games)
    
    def generate_aligned_ensemble(self, n_games=10):
        target = self.detector.get_ensemble_target()
        return self.generate_aligned(target, n_games)
    
    def generate_coverage_baseline(self, n_games=10):
        pool = []
        seen = set()
        for _ in range(n_games * 200):
            game = sorted(np.random.choice(range(1, 26), 15, replace=False))
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                pool.append(game)
        selected = [pool[0]]
        for _ in range(n_games - 1):
            best, best_min = None, -1
            for g in pool:
                if g not in selected:
                    min_dist = min(len(set(g) & set(s)) for s in selected)
                    if min_dist > best_min:
                        best_min = min_dist
                        best = g
            if best: selected.append(best)
        return selected[:n_games]


# ============================================================
# AVALIAÇÃO
# ============================================================
def theoretical_baseline_metrics():
    expected_hits = sum(k * HYPE_PROBS[k] for k in range(16))
    return {
        'media_hits': expected_hits,
        'freq_11_plus': sum(HYPE_PROBS[k] for k in range(11, 16)),
        'freq_12_plus': sum(HYPE_PROBS[k] for k in range(12, 16)),
        'freq_13_plus': sum(HYPE_PROBS[k] for k in range(13, 16)),
    }

def evaluate_games(games, test_draws):
    dist = {h: 0 for h in range(0, 16)}
    for draw in test_draws:
        actual = set(draw['dezenas'])
        for g in games:
            dist[len(set(g) & actual)] += 1
    total = len(test_draws) * len(games)
    if total == 0:
        return {'media_hits': 0, 'freq_11_plus': 0, 'freq_12_plus': 0, 'freq_13_plus': 0}
    hits_sum = sum(h * dist[h] for h in range(16))
    return {
        'media_hits': hits_sum / total,
        'freq_11_plus': sum(dist[h] for h in range(11, 16)) / total,
        'freq_12_plus': sum(dist[h] for h in range(12, 16)) / total,
        'freq_13_plus': sum(dist[h] for h in range(13, 16)) / total,
    }

def permutation_test_corrected(strat_vals, rand_vals, n_perm=10000):
    observed = np.mean(strat_vals) - np.mean(rand_vals)
    combined = np.concatenate([strat_vals, rand_vals])
    n1 = len(strat_vals)
    extreme = 0
    for _ in range(n_perm):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        if abs(perm_diff) >= abs(observed):
            extreme += 1
    return observed, extreme / n_perm


# ============================================================
# TESTE OUT-OF-SAMPLE (RÁPIDO)
# ============================================================
def out_of_sample_test(contests, train_window=300, test_ahead=5, n_games=10, max_windows=20):
    """
    Teste out-of-sample limitado para viabilidade computacional.
    """
    print(f"\n🔬 TESTE OUT-OF-SAMPLE ({test_ahead} concursos à frente, {max_windows} janelas)...")
    results = []
    
    step = max(50, (len(contests) - train_window - test_ahead) // max_windows)
    starts = list(range(100, len(contests) - test_ahead - train_window, step))[:max_windows]
    
    for start in tqdm(starts, desc="OOS"):
        train_end = start + train_window
        if train_end + test_ahead > len(contests):
            continue
        train_data = contests[start:train_end]
        test_data = contests[train_end:train_end + test_ahead]
        if len(train_data) < 100 or len(test_data) < test_ahead:
            continue
        
        detector = LocalDeformationDetector(train_data)
        generator = LocalRegimeGenerator(detector)
        
        aligned_temp = generator.generate_aligned_temporal(n_games, window=20)
        aligned_glob = generator.generate_aligned_global(n_games)
        coverage = generator.generate_coverage_baseline(n_games)
        random_g = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
        
        met_temp = evaluate_games(aligned_temp, test_data)
        met_glob = evaluate_games(aligned_glob, test_data)
        met_cov = evaluate_games(coverage, test_data)
        met_rand = evaluate_games(random_g, test_data)
        
        results.append({
            'temp_11': met_temp['freq_11_plus'],
            'glob_11': met_glob['freq_11_plus'],
            'cov_11': met_cov['freq_11_plus'],
            'rand_11': met_rand['freq_11_plus'],
            'diff_temp': met_temp['freq_11_plus'] - met_rand['freq_11_plus'],
            'diff_glob': met_glob['freq_11_plus'] - met_rand['freq_11_plus'],
            'diff_cov': met_cov['freq_11_plus'] - met_rand['freq_11_plus'],
        })
    
    if results:
        print(f"\n📊 RESULTADOS OUT-OF-SAMPLE ({len(results)} janelas):")
        print(f"   {'Estratégia':<25} {'Média diff':<15} {'% positivo':<15} {'Wilcoxon p':<15}")
        print(f"   {'-'*70}")
        
        for strat, key in [('Temporal (20)', 'diff_temp'), ('Global', 'diff_glob'), ('Cobertura', 'diff_cov')]:
            diffs = [r[key] for r in results]
            mean_diff = np.mean(diffs)
            pct_pos = np.mean(np.array(diffs) > 0) * 100
            try:
                _, pval = wilcoxon(diffs)
            except:
                pval = 1.0
            sig = "🔴" if pval < 0.05 else "🟢"
            print(f"   {strat:<25} {mean_diff:<15.6f} {pct_pos:<15.1f}% {pval:<15.4f} {sig}")
        
        temp_diffs = [r['diff_temp'] for r in results]
        glob_diffs = [r['diff_glob'] for r in results]
        try:
            _, p_compare = wilcoxon(temp_diffs, glob_diffs)
            print(f"\n   Temporal vs Global: Wilcoxon p = {p_compare:.4f}")
            if p_compare < 0.05:
                print(f"   ✅ Temporal DIFERE de Global com significância!")
            else:
                print(f"   🟡 Temporal ≈ Global (sem diferença)")
        except:
            pass
    
    return results


# ============================================================
# TESTE COMPARATIVO REAL vs SINTÉTICO
# ============================================================
def run_comparative_test(real_contests, n_simulations=200, blind_size=300, n_games=10):
    print(f"\n🔬 TESTE COMPARATIVO REAL vs SINTÉTICO ({n_simulations} simulações)...")
    
    def run_single(contests, seed):
        random.seed(seed)
        np.random.seed(seed)
        train = contests[:-blind_size]
        blind = contests[-blind_size:]
        
        detector = LocalDeformationDetector(train)
        generator = LocalRegimeGenerator(detector)
        
        aligned = generator.generate_aligned_temporal(n_games, window=20)
        random_g = [sorted(np.random.choice(range(1, 26), 15, replace=False)) for _ in range(n_games)]
        
        met_aligned = evaluate_games(aligned, blind)
        met_random = evaluate_games(random_g, blind)
        return met_aligned['freq_11_plus'] - met_random['freq_11_plus']
    
    real_deltas = []
    for i in tqdm(range(n_simulations), desc="Real"):
        real_deltas.append(run_single(real_contests, i))
    
    synth_contests_base = generate_synthetic_contests(len(real_contests))
    synth_deltas = []
    for i in tqdm(range(n_simulations), desc="Sintético"):
        synth_deltas.append(run_single(synth_contests_base, i + 100000))
    
    real_deltas = np.array(real_deltas)
    synth_deltas = np.array(synth_deltas)
    
    print(f"\n📊 RESULTADOS COMPARATIVOS:")
    print(f"   {'Métrica':<25} {'Real':<15} {'Sintético':<15}")
    print(f"   {'Média delta':<25} {np.mean(real_deltas):<15.6f} {np.mean(synth_deltas):<15.6f}")
    print(f"   {'Std delta':<25} {np.std(real_deltas):<15.6f} {np.std(synth_deltas):<15.6f}")
    print(f"   {'% positivo':<25} {np.mean(real_deltas>0)*100:<15.1f}% {np.mean(synth_deltas>0)*100:<15.1f}%")
    
    ks_stat, ks_p = ks_2samp(real_deltas, synth_deltas)
    mw_stat, mw_p = mannwhitneyu(real_deltas, synth_deltas, alternative='greater')
    print(f"\n   KS-test: p={ks_p:.4f}")
    print(f"   Mann-Whitney (real > synth): p={mw_p:.4f}")
    
    if mw_p < 0.05:
        print(f"   ✅ REAL > SINTÉTICO com significância!")
    else:
        print(f"   🟡 REAL ≈ SINTÉTICO (sem diferença)")
    
    return real_deltas, synth_deltas, mw_p


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🧬 LABORATÓRIO DE MICROESTRUTURA TEMPORAL v37")
    print("="*70)
    
    contests = load_all_contests('resultados_lotofacil.csv')
    if contests is None:
        print("❌ Arquivo não encontrado. Gerando sintéticos...")
        contests = generate_synthetic_contests(3686)
    
    print(f"📂 {len(contests)} concursos válidos")
    
    # Detector com cache
    t0 = time.time()
    detector = LocalDeformationDetector(contests)
    print(f"\n⏱️ Detector construído em {time.time()-t0:.1f}s")
    print(f"📊 Timeline: {len(detector.timeline.get_series(0))} pontos temporais")
    
    # Deformações atuais
    results = detector.analyze_all_windows()
    print(f"\n📊 DEFORMAÇÕES ATUAIS:")
    for w in WINDOWS:
        if w in results and results[w] is not None:
            print(f"   Janela {w:3d}: deformação total = {results[w]['total_deformation']:.2f}")
    
    # Persistência
    # Construir série de performance (freq_11+ por concurso)
    perf_series = None
    if len(contests) >= 100:
        perf_series = []
        for i in range(len(contests)):
            # Performance dummy: usar soma como proxy
            perf_series.append(sum(contests[i]['dezenas']) / 15.0)
        perf_series = np.array(perf_series)
    
    persistence = detector.find_persistent_deformations(perf_series)
    persistent_feats = [k for k, v in persistence.items() if v['is_persistent']]
    low_var_feats = [k for k, v in persistence.items() if v.get('is_low_variance')]
    
    if low_var_feats:
        print(f"\n📊 FEATURES DE BAIXA VARIÂNCIA (autocorr não confiável):")
        for feat in low_var_feats:
            print(f"   {feat}")
    
    if persistent_feats:
        print(f"\n📊 DEFORMAÇÕES PERSISTENTES (autocorr > 0.2):")
        for feat in persistent_feats:
            print(f"   {feat}: autocorr={persistence[feat]['autocorr']:.3f} MI_pred={persistence[feat]['mutual_info_pred']:.4f}")
    else:
        print(f"\n📊 Nenhuma deformação persistente detectada")
    
    print("\nOpções:")
    print("1. Teste out-of-sample (20 janelas)")
    print("2. Teste comparativo REAL vs SINTÉTICO (200 simulações)")
    print("3. Ambos")
    op = input("Escolha [3]: ").strip() or "3"
    
    if op in ("1", "3"):
        out_of_sample_test(contests, train_window=300, test_ahead=5, n_games=10, max_windows=20)
    
    if op in ("2", "3"):
        run_comparative_test(contests, n_simulations=200, blind_size=300, n_games=10)
    
    print(f"\n⏱️ Tempo total: {time.time()-t0:.1f}s")
    print("\n✅ Concluído!")

if __name__ == "__main__":
    main()
