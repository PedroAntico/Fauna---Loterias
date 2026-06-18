#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v73.1
CORREÇÃO DO SINAL DO CICLO + PAINEL DE PRESSÃO POR ESTADO

CORREÇÕES vs v73:
✅ Score de ciclo agora avalia o valor do JOGO CANDIDATO (não o estado atual)
✅ Sinal corrigido: z positivo (atrasado) → reduz score (bônus)
✅ Painel de pressão mostra z‑score para TODOS os valores possíveis
✅ Mantém as 7 camadas do v73 (centro, persistência, ciclo, correlação, transição, linhas, blocos)
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter, defaultdict
from itertools import combinations
import os, random, time, warnings, json
from math import comb
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTES
# ============================================================
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}
MOLDURA = {1,2,3,4,5, 6,10, 11,15, 16,20, 21,22,23,24,25}
HYPE_PROBS = {k: hypergeom.pmf(k, 25, 15, 15) for k in range(0, 16)}
PREMIO_VALORES = {11: 6.0, 12: 12.0, 13: 30.0, 14: 1500.0, 15: 1800000.0}
CUSTO_APOSTA = 3.5

STRUCTURAL_PARAMS = ['pares', 'moldura', 'primos', 'repeticoes', 'amplitude']

LINHAS = {
    1: [1, 2, 3, 4, 5],
    2: [6, 7, 8, 9, 10],
    3: [11, 12, 13, 14, 15],
    4: [16, 17, 18, 19, 20],
    5: [21, 22, 23, 24, 25]
}
BLOCOS = {
    'B1 (01-05)': [1, 2, 3, 4, 5],
    'B2 (06-10)': [6, 7, 8, 9, 10],
    'B3 (11-15)': [11, 12, 13, 14, 15],
    'B4 (16-20)': [16, 17, 18, 19, 20],
    'B5 (21-25)': [21, 22, 23, 24, 25]
}

# ============================================================
# BITMASK
# ============================================================
class BitmaskCache:
    def __init__(self):
        self._cache = {}
    def get_mask(self, game):
        key = tuple(game) if isinstance(game, list) else game
        if key not in self._cache:
            mask = 0
            for d in key:
                mask |= (1 << d)
            self._cache[key] = mask
        return self._cache[key]

BITMASK_CACHE = BitmaskCache()
mask_intersection = lambda m1, m2: (m1 & m2).bit_count()

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
# GERADOR DE JOGOS
# ============================================================
class LooseGenerator:
    def __init__(self):
        pass

    def generate_one(self, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None):
        for _ in range(500):
            game = self._generate_raw(fixed, semifixed, min_semifixed, max_semifixed)
            if game is not None:
                return game
        raise RuntimeError("Não foi possível gerar jogo com os parâmetros fornecidos.")

    def _generate_raw(self, fixed, semifixed, min_semifixed, max_semifixed):
        if fixed is None: fixed = []
        if semifixed is None: semifixed = []
        fixed_set = set(fixed)
        semifixed_set = set(semifixed) - fixed_set
        proibidas = fixed_set | semifixed_set
        restantes = list(set(range(1, 26)) - proibidas)
        n_fixas = len(fixed_set)
        max_semi = min(max_semifixed, len(semifixed_set)) if max_semifixed is not None else len(semifixed_set)
        min_semi = max(min_semifixed, 0)
        if min_semi > max_semi:
            return None
        n_semifixed_escolher = random.randint(min_semi, max_semi)
        n_restantes = 15 - n_fixas - n_semifixed_escolher
        if n_restantes < 0 or n_restantes > len(restantes):
            return None
        for _ in range(200):
            chosen_semi = set(random.sample(list(semifixed_set), min(n_semifixed_escolher, len(semifixed_set)))) if n_semifixed_escolher > 0 and len(semifixed_set) > 0 else set()
            chosen_rest = set(random.sample(restantes, min(n_restantes, len(restantes)))) if n_restantes > 0 else set()
            game = sorted(fixed_set | chosen_semi | chosen_rest)
            if len(game) == 15:
                return game
        return None

    def generate_pure_random(self):
        return sorted(np.random.choice(range(1, 26), 15, replace=False))

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================
def extract_filter(dezenas, filter_name, prev_dezenas=None):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'amplitude': return max(d) - min(d)
    if filter_name == 'repeticoes':
        if prev_dezenas is None: return 8
        return len(set(d) & set(prev_dezenas))
    return 0

def count_in_line(dezenas, linha_num):
    linha_dezenas = set(LINHAS[linha_num])
    return len(set(dezenas) & linha_dezenas)

def count_in_block(dezenas, block_dezenas):
    return len(set(dezenas) & set(block_dezenas))

# ============================================================
# STRUCTURAL PREDICTOR
# ============================================================
class StructuralPredictorV731:
    def __init__(self, contests, cache_file='v731_weights_cache.json'):
        self.contests = contests
        self.active_filters = STRUCTURAL_PARAMS
        self.windows = [20, 50, 100, 200]
        self.cache_file = cache_file
        self._load_or_compute_weights()

    def _get_filter_series(self, filter_name):
        series = []
        for i, c in enumerate(self.contests):
            prev = self.contests[i-1]['dezenas'] if i > 0 else None
            series.append(extract_filter(c['dezenas'], filter_name, prev))
        return np.array(series, dtype=float)

    def _load_or_compute_weights(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                self.weights = data['weights']
                self.gains = data['gains']
                return
            except:
                pass
        self._compute_real_gains()
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'weights': self.weights, 'gains': self.gains}, f)
        except:
            pass

    def _compute_real_gains(self, min_history=500):
        gains = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            acertos_pred, acertos_base = 0, 0
            total = 0
            for t in range(min_history, len(self.contests) - 1):
                history = series[:t+1]
                next_val = series[t+1]
                curr_val = series[t]
                preds = []
                for w in self.windows:
                    if len(history) >= w:
                        preds.append(np.median(history[-w:]))
                if preds:
                    center = np.median(preds)
                    if int(round(center)) == int(next_val):
                        acertos_pred += 1
                if int(curr_val) == int(next_val):
                    acertos_base += 1
                total += 1
            gains[filtro] = (acertos_pred / total - acertos_base / total) * 100 if total > 0 else 0.0
        self.gains = gains
        total_gain = sum(max(0, g) for g in gains.values())
        if total_gain > 0:
            self.weights = {f: max(0, gains[f]) / total_gain for f in gains}
        else:
            self.weights = {f: 1.0/len(gains) for f in gains}

    def predict_centers(self):
        centers = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            preds = []
            for w in self.windows:
                if len(series) >= w:
                    preds.append(np.median(series[-w:]))
            if preds:
                centers[filtro] = np.median(preds)
        return centers

# ============================================================
# MODELO DE PRESSÃO ESTRUTURAL TEMPORAL CORRIGIDO (v73.1)
# ============================================================
class PressaoEstruturalTemporalV731:
    def __init__(self, contests, predictor_weights=None):
        self.contests = contests
        self.predictor_weights = predictor_weights if predictor_weights else {p: 0.2 for p in STRUCTURAL_PARAMS}
        self._build_historical_stats()

    def _extract_param_series(self, param_name):
        series = []
        for i, c in enumerate(self.contests):
            prev = self.contests[i-1]['dezenas'] if i > 0 else None
            series.append(extract_filter(c['dezenas'], param_name, prev))
        return np.array(series, dtype=int)

    def _compute_stats_for_series(self, series, name):
        stats = {}
        s = np.array(series, dtype=int)
        stats['series'] = s
        stats['freq'] = Counter(s)
        stats['intervalo_medio'] = {}
        stats['desvio_intervalo'] = {}
        stats['atraso_atual'] = {}
        stats['z_score_atraso'] = {}
        for val in set(s):
            ocorrencias = np.where(s == val)[0]
            if len(ocorrencias) > 1:
                intervalos = np.diff(ocorrencias)
                stats['intervalo_medio'][val] = np.mean(intervalos)
                stats['desvio_intervalo'][val] = np.std(intervalos)
            else:
                stats['intervalo_medio'][val] = len(s)
                stats['desvio_intervalo'][val] = len(s)
            if len(ocorrencias) > 0:
                atraso = len(s) - 1 - ocorrencias[-1]
                stats['atraso_atual'][val] = atraso
                intervalo = stats['intervalo_medio'][val]
                desvio = stats['desvio_intervalo'][val]
                if desvio > 0:
                    stats['z_score_atraso'][val] = (atraso - intervalo) / desvio
                else:
                    stats['z_score_atraso'][val] = 0.0
            else:
                stats['atraso_atual'][val] = len(s)
                stats['z_score_atraso'][val] = 0.0
        runs = []
        if len(s) > 0:
            current_val = s[0]
            current_len = 1
            for i in range(1, len(s)):
                if s[i] == current_val:
                    current_len += 1
                else:
                    runs.append((current_val, current_len))
                    current_val = s[i]
                    current_len = 1
            runs.append((current_val, current_len))
        stats['persistencia_atual'] = {}
        stats['persistencia_media'] = {}
        if runs:
            last_val, last_len = runs[-1]
            stats['persistencia_atual'][last_val] = last_len
        for val in set(s):
            lengths = [l for v, l in runs if v == val]
            if lengths:
                stats['persistencia_media'][val] = np.mean(lengths)
            else:
                stats['persistencia_media'][val] = 1.0
        return stats

    def _build_historical_stats(self):
        self.stats = {}
        for param in STRUCTURAL_PARAMS:
            s = self._extract_param_series(param)
            self.stats[param] = self._compute_stats_for_series(s, param)
        self.stats['linhas'] = {}
        for linha_num in range(1, 6):
            s = [count_in_line(c['dezenas'], linha_num) for c in self.contests]
            self.stats['linhas'][linha_num] = self._compute_stats_for_series(s, f'linha_{linha_num}')
        self.stats['blocos'] = {}
        for bloco_nome, bloco_dezenas in BLOCOS.items():
            s = [count_in_block(c['dezenas'], bloco_dezenas) for c in self.contests]
            self.stats['blocos'][bloco_nome] = self._compute_stats_for_series(s, bloco_nome)

    def _compute_game_features(self, game, prev_dezenas):
        features = {}
        for param in STRUCTURAL_PARAMS:
            features[param] = extract_filter(game, param, prev_dezenas)
        for linha_num in range(1, 6):
            features[f'linha_{linha_num}'] = count_in_line(game, linha_num)
        for bloco_nome in BLOCOS:
            features[bloco_nome] = count_in_block(game, BLOCOS[bloco_nome])
        return features

    # Camada 1: Centro
    def score_centro(self, game_features, centers):
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features and param in centers:
                if param == 'amplitude':
                    dist = abs(game_features[param] - centers[param]) / 14.0
                else:
                    dist = abs(game_features[param] - centers[param])
                score += self.predictor_weights.get(param, 0.2) * dist
        return score

    # Camada 2: Persistência
    def score_persistencia(self, game_features):
        score = 0.0
        count = 0
        for param in STRUCTURAL_PARAMS:
            if param in game_features and param in self.stats:
                val = game_features[param]
                pers_atual = self.stats[param]['persistencia_atual'].get(val, 1)
                pers_media = self.stats[param]['persistencia_media'].get(val, 1.0)
                if pers_media > 0:
                    score += (pers_atual / pers_media - 1.0) * self.predictor_weights.get(param, 0.2)
                    count += 1
        return score / max(1, count)

    # Camada 3: Ciclo CORRIGIDO
    def score_ciclo(self, game_features):
        """
        Avalia o atraso do valor que o JOGO CANDIDATO está propondo.
        Z positivo (atrasado) → reduz o score (bônus).
        Z negativo (recente) → aumenta o score (penalidade).
        """
        score = 0.0
        count = 0
        # Parâmetros estruturais
        for param in STRUCTURAL_PARAMS:
            if param in game_features and param in self.stats:
                val = game_features[param]  # valor do jogo candidato
                z = self.stats[param]['z_score_atraso'].get(val, 0.0)
                # SUBTRAI: z positivo (atrasado) → score menor (melhor)
                score -= z * self.predictor_weights.get(param, 0.2)
                count += 1
        # Linhas
        for linha_num in range(1, 6):
            key = f'linha_{linha_num}'
            if key in game_features and 'linhas' in self.stats and linha_num in self.stats['linhas']:
                val = game_features[key]
                z = self.stats['linhas'][linha_num]['z_score_atraso'].get(val, 0.0)
                score -= z * 0.15
                count += 1
        # Blocos
        for bloco_nome in BLOCOS:
            if bloco_nome in game_features and 'blocos' in self.stats and bloco_nome in self.stats['blocos']:
                val = game_features[bloco_nome]
                z = self.stats['blocos'][bloco_nome]['z_score_atraso'].get(val, 0.0)
                score -= z * 0.15
                count += 1
        return score / max(1, count)

    # Camada 4: Correlação
    def score_correlacao(self, game_features):
        score = 0.0
        pairs = [('moldura', 'pares'), ('primos', 'pares'), ('repeticoes', 'moldura')]
        for p1, p2 in pairs:
            if p1 in game_features and p2 in game_features:
                v1, v2 = game_features[p1], game_features[p2]
                recent = self.contests[-200:] if len(self.contests) >= 200 else self.contests
                joint_count = 0
                total_with_v1 = 0
                for i, c in enumerate(recent):
                    idx = len(self.contests) - len(recent) + i
                    prev = self.contests[idx-1]['dezenas'] if idx > 0 else None
                    f1 = extract_filter(c['dezenas'], p1, prev)
                    f2 = extract_filter(c['dezenas'], p2, prev)
                    if f1 == v1:
                        total_with_v1 += 1
                        if f2 == v2:
                            joint_count += 1
                if total_with_v1 > 0:
                    prob = joint_count / total_with_v1
                    if prob < 0.2:
                        score += (0.2 - prob) * 5.0
        return score

    # Camada 5: Transição
    def score_transicao(self, game_features):
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features and param in self.stats:
                s = self.stats[param]['series']
                if len(s) >= 2:
                    prev_val = s[-1]
                    curr_val = game_features[param]
                    trans_count = 0
                    total_prev = 0
                    for i in range(1, len(s)):
                        if s[i-1] == prev_val:
                            total_prev += 1
                            if s[i] == curr_val:
                                trans_count += 1
                    if total_prev > 0:
                        prob = trans_count / total_prev
                        if prob < 0.15:
                            score += (0.15 - prob) * 5.0
        return score

    # Score global
    def compute_global_score(self, game_features, centers):
        s_centro = self.score_centro(game_features, centers)
        s_persist = self.score_persistencia(game_features)
        s_ciclo = self.score_ciclo(game_features)
        s_correl = self.score_correlacao(game_features)
        s_trans = self.score_transicao(game_features)
        return (0.25 * s_centro + 0.20 * s_persist + 0.25 * s_ciclo +
                0.10 * s_correl + 0.10 * s_trans)

    # Painel de pressão expandido
    def display_current_pressure(self):
        """Exibe a pressão (z‑score) para TODOS os valores possíveis de cada parâmetro, linha e bloco."""
        print(f"\n📊 PAINEL DE PRESSÃO POR ESTADO (z‑score do atraso)")
        print(f"   Valores com z > 0 = atrasados (favorecidos pelo modelo)")
        print(f"   Valores com z < 0 = recentes (penalizados pelo modelo)\n")
        
        for param in STRUCTURAL_PARAMS:
            print(f"   --- {param} ---")
            z_scores = self.stats[param]['z_score_atraso']
            for val in sorted(z_scores.keys()):
                z = z_scores[val]
                bar = "█" * int(max(0, z)) if z > 0 else "░" * int(max(0, -z))
                print(f"   {val:3d}: z={z:+6.2f} {bar}")
            print()
        
        print(f"   --- LINHAS ---")
        for linha_num in range(1, 6):
            print(f"   Linha {linha_num} ({LINHAS[linha_num]}):")
            z_scores = self.stats['linhas'][linha_num]['z_score_atraso']
            for val in sorted(z_scores.keys()):
                z = z_scores[val]
                bar = "█" * int(max(0, z)) if z > 0 else "░" * int(max(0, -z))
                print(f"   {val:3d}: z={z:+6.2f} {bar}")
            print()
        
        print(f"   --- BLOCOS ---")
        for bloco_nome in BLOCOS:
            print(f"   {bloco_nome}:")
            z_scores = self.stats['blocos'][bloco_nome]['z_score_atraso']
            for val in sorted(z_scores.keys()):
                z = z_scores[val]
                bar = "█" * int(max(0, z)) if z > 0 else "░" * int(max(0, -z))
                print(f"   {val:3d}: z={z:+6.2f} {bar}")
            print()

# ============================================================
# OTIMIZADOR DE CARTEIRA v73.1
# ============================================================
class PortfolioOptimizerV731:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 use_pressure_model=False):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.predictor = StructuralPredictorV731(contests)
        self.use_pressure_model = use_pressure_model
        if use_pressure_model:
            self.pressure = PressaoEstruturalTemporalV731(contests, self.predictor.weights)
        else:
            self.pressure = None

    def _score_game(self, game, centers, prev_dezenas):
        features = {}
        for param in STRUCTURAL_PARAMS:
            features[param] = extract_filter(game, param, prev_dezenas)
        for linha_num in range(1, 6):
            features[f'linha_{linha_num}'] = count_in_line(game, linha_num)
        for bloco_nome in BLOCOS:
            features[bloco_nome] = count_in_block(game, BLOCOS[bloco_nome])
        if self.use_pressure_model and self.pressure is not None:
            return self.pressure.compute_global_score(features, centers)
        else:
            score = 0.0
            for filtro, center in centers.items():
                if filtro in features and filtro in self.predictor.weights:
                    if filtro == 'amplitude':
                        dist = abs(features[filtro] - center) / 14.0
                    else:
                        dist = abs(features[filtro] - center)
                    score += self.predictor.weights.get(filtro, 0.2) * dist
            return score

    def generate_pool(self, n_candidates, prev_dezenas=None):
        pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool", leave=False):
            try:
                g = self.generator.generate_one(
                    fixed=self.fixed, semifixed=self.semifixed,
                    min_semifixed=self.min_semifixed, max_semifixed=self.max_semifixed)
                key = tuple(g)
                if key not in seen:
                    seen.add(key)
                    pool.append(g)
            except RuntimeError:
                break
        return pool

    def optimize(self, n_games=5, n_candidates=50000, n_central=2, n_intermed=2, n_perif=1):
        centers = self.predictor.predict_centers()
        prev_dezenas = self.contests[-1]['dezenas'] if self.contests else None
        pool = self.generate_pool(n_candidates, prev_dezenas)
        if len(pool) < n_games:
            return []
        scored = [(self._score_game(g, centers, prev_dezenas), g) for g in pool]
        scored.sort(key=lambda x: x[0])
        n_total = len(scored)
        idx1 = min(n_central * n_total // n_games, n_total)
        idx2 = min((n_central + n_intermed) * n_total // n_games, n_total)
        grupo_central = scored[:idx1]
        grupo_intermed = scored[idx1:idx2]
        grupo_perif = scored[idx2:]
        def select_diverse(group, n_select):
            if len(group) <= n_select:
                return [g for _, g in group[:n_select]]
            return [g for _, g in group[:n_select]]
        centrais = select_diverse(grupo_central, n_central)
        intermed = select_diverse(grupo_intermed, n_intermed)
        perifs = select_diverse(grupo_perif, n_perif)
        combined = centrais + intermed + perifs
        return combined[:n_games]

    def backtest(self, portfolio, test_draws):
        if len(portfolio) == 0:
            return {'lift': 0, 'roi': 0, 'hit_distribution': {k:0 for k in range(11,16)}}
        n_success = total_premio = 0
        total_custo = len(portfolio) * len(test_draws) * CUSTO_APOSTA
        portfolio_masks = np.array([BITMASK_CACHE.get_mask(g) for g in portfolio], dtype=np.uint32)
        hit_counts = {k:0 for k in range(11,16)}
        for draw in test_draws:
            dm = BITMASK_CACHE.get_mask(draw['dezenas'])
            for pm in portfolio_masks:
                hits = mask_intersection(pm, dm)
                if hits >= 11:
                    n_success += 1
                    total_premio += PREMIO_VALORES.get(hits, 0)
                    hit_counts[hits] += 1
        prob = n_success/(len(portfolio)*len(test_draws)) if test_draws else 0
        p_single = sum(HYPE_PROBS[k] for k in range(11,16))
        theo_prob = 1 - (1-p_single)**len(portfolio)
        return {'empirical': prob, 'theoretical': theo_prob,
                'lift': prob/theo_prob if theo_prob>0 else 1.0,
                'n_test': len(test_draws), 'n_success': n_success,
                'total_premio': total_premio, 'total_custo': total_custo,
                'roi': (total_premio-total_custo)/total_custo*100 if total_custo>0 else 0,
                'hit_distribution': hit_counts}

# ============================================================
# WALK‑FORWARD COMPARATIVO
# ============================================================
def walk_forward_compare(contests, train_size=500, step=50):
    print(f"\n🔬 WALK‑FORWARD: MODELO PLANO vs MODELO DE PRESSÃO (v73.1)")
    print(f"   Treino: {train_size} | Teste: {step}\n")
    results = {'plano': [], 'pressao': []}
    start = train_size
    while start + step <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+step]
        for modelo, use_press in [('plano', False), ('pressao', True)]:
            try:
                opt = PortfolioOptimizerV731(train_data, use_pressure_model=use_press)
                portfolio = opt.optimize(5, 20000, 2, 2, 1)
                bt = opt.backtest(portfolio, test_data)
                results[modelo].append({
                    'lift': bt['lift'],
                    'roi': bt['roi'],
                    '13pts': bt['hit_distribution'].get(13, 0),
                    '14pts': bt['hit_distribution'].get(14, 0),
                })
            except Exception as e:
                results[modelo].append({'lift': 0, 'roi': 0, '13pts': 0, '14pts': 0})
        l_plano = results['plano'][-1]['lift'] if results['plano'] else 0
        l_press = results['pressao'][-1]['lift'] if results['pressao'] else 0
        print(f"   Janela {start}: plano(lift={l_plano:.3f}) | pressão(lift={l_press:.3f})")
        start += step
    print(f"\n📊 RESULTADO FINAL:")
    for modelo in ['plano', 'pressao']:
        avg_lift = np.mean([r['lift'] for r in results[modelo]]) if results[modelo] else 0
        avg_roi = np.mean([r['roi'] for r in results[modelo]]) if results[modelo] else 0
        total_13 = sum(r['13pts'] for r in results[modelo])
        total_14 = sum(r['14pts'] for r in results[modelo])
        nome = "Modelo plano" if modelo == 'plano' else "Modelo de pressão"
        print(f"   {nome}:")
        print(f"      Média lift: {avg_lift:.3f} | Média ROI: {avg_roi:.1f}%")
        print(f"      Total 13pts: {total_13} | 14pts: {total_14}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v73.1")
    print("   CORREÇÃO DO SINAL DO CICLO + PAINEL DE PRESSÃO POR ESTADO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Walk‑forward comparativo: modelo plano vs modelo de pressão")
        print("2. Gerar carteira com modelo de pressão")
        print("3. Painel de pressão por estado (todos os valores possíveis)")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
            walk_forward_compare(contests)

        elif op == '2':
            print("\n📝 CONFIGURAÇÃO DA CARTEIRA")
            fixed_str = input("   Dezenas fixas (ex: 15 16 20 ou ENTER): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            semifixed_str = input("   Dezenas semifixas (ex: 03 07 14 25 ou ENTER): ").strip()
            semifixed = [int(x) for x in semifixed_str.split()] if semifixed_str else []
            min_semi, max_semi = 0, None
            if semifixed:
                try:
                    min_semi = int(input(f"   Mínimo de semifixas [0-{len(semifixed)}]: ").strip() or "0")
                    max_semi = int(input(f"   Máximo de semifixas [0-{len(semifixed)}]: ").strip() or str(len(semifixed)))
                except:
                    min_semi, max_semi = 0, len(semifixed)
            opt = PortfolioOptimizerV731(contests, fixed=fixed, semifixed=semifixed,
                                         min_semifixed=min_semi, max_semifixed=max_semi,
                                         use_pressure_model=True)
            portfolio = opt.optimize(5, 50000, 2, 2, 1)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                rep = len(set(g) & set(contests[-1]['dezenas'])) if contests else 0
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")

        elif op == '3':
            predictor = StructuralPredictorV731(contests)
            pressure = PressaoEstruturalTemporalV731(contests, predictor.weights)
            pressure.display_current_pressure()

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
