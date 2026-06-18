#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v72
MODELO DE PRESSÃO ESTRUTURAL TEMPORAL (5 CAMADAS)

CAMADAS DO SCORE:
✅ Centro estrutural (distância ao centro previsto)
✅ Persistência (duração da sequência atual vs. média histórica)
✅ Ciclo (atraso atual vs. intervalo médio de ocorrência)
✅ Correlação condicional (distribuição de pares dado moldura, etc.)
✅ Matriz de transição (Markov: probabilidade do próximo estado dado o atual)
✅ Score global = combinação ponderada das 5 camadas
✅ Walk‑forward comparativo: modelo plano (v71) vs. modelo de pressão (v72)
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

# Parâmetros estruturais monitorados
STRUCTURAL_PARAMS = ['pares', 'moldura', 'primos', 'repeticoes', 'amplitude']

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

# ============================================================
# MODELO DE PRESSÃO ESTRUTURAL TEMPORAL (5 CAMADAS)
# ============================================================
class PressaoEstruturalTemporal:
    def __init__(self, contests):
        self.contests = contests
        self._build_historical_stats()

    def _extract_param_series(self, param_name):
        """Extrai a série temporal de um parâmetro estrutural."""
        series = []
        for i, c in enumerate(self.contests):
            prev = self.contests[i-1]['dezenas'] if i > 0 else None
            series.append(extract_filter(c['dezenas'], param_name, prev))
        return np.array(series, dtype=int)

    def _build_historical_stats(self):
        """Pré‑computa estatísticas históricas para todos os parâmetros."""
        self.series = {}
        self.freq = {}
        self.intervalo_medio = {}
        self.desvio_intervalo = {}
        self.atraso_atual = {}
        self.persistencia_atual = {}
        self.persistencia_media = {}
        for param in STRUCTURAL_PARAMS:
            s = self._extract_param_series(param)
            self.series[param] = s
            # Frequência de cada valor
            self.freq[param] = Counter(s)
            # Intervalo entre ocorrências de cada valor
            self.intervalo_medio[param] = {}
            self.desvio_intervalo[param] = {}
            self.atraso_atual[param] = {}
            for val in set(s):
                ocorrencias = np.where(s == val)[0]
                if len(ocorrencias) > 1:
                    intervalos = np.diff(ocorrencias)
                    self.intervalo_medio[param][val] = np.mean(intervalos)
                    self.desvio_intervalo[param][val] = np.std(intervalos)
                else:
                    self.intervalo_medio[param][val] = len(s)
                    self.desvio_intervalo[param][val] = len(s)
                # Atraso atual
                if len(ocorrencias) > 0:
                    self.atraso_atual[param][val] = len(s) - 1 - ocorrencias[-1]
                else:
                    self.atraso_atual[param][val] = len(s)
            # Persistência (sequências)
            self.persistencia_atual[param] = {}
            self.persistencia_media[param] = {}
            runs = []
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
            # Persistência atual = duração da sequência mais recente
            last_val, last_len = runs[-1]
            self.persistencia_atual[param][last_val] = last_len
            # Persistência média por valor
            for val in set(s):
                lengths = [l for v, l in runs if v == val]
                if lengths:
                    self.persistencia_media[param][val] = np.mean(lengths)
                else:
                    self.persistencia_media[param][val] = 1.0

    # Camada 1: Centro estrutural
    def score_centro(self, game_features, centers, weights):
        """Distância ponderada aos centros previstos."""
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features and param in centers:
                if param == 'amplitude':
                    dist = abs(game_features[param] - centers[param]) / 14.0
                else:
                    dist = abs(game_features[param] - centers[param])
                score += weights.get(param, 0.2) * dist
        return score

    # Camada 2: Persistência
    def score_persistencia(self, game_features):
        """
        Mede se a duração atual de um valor está acima da média histórica.
        Persistência alta → o sistema pode "cansar" e reverter.
        Score positivo = persistência acima da média (pressionado para cair).
        """
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features:
                val = game_features[param]
                pers_atual = self.persistencia_atual[param].get(val, 1)
                pers_media = self.persistencia_media[param].get(val, 1.0)
                if pers_media > 0:
                    score += (pers_atual / pers_media - 1.0)  # >0 se acima da média
        return score / len(STRUCTURAL_PARAMS)

    # Camada 3: Ciclo
    def score_ciclo(self, game_features):
        """
        Mede o atraso atual em relação ao intervalo médio de ocorrência.
        Valores muito atrasados recebem pressão positiva (devem voltar).
        """
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features:
                val = game_features[param]
                atraso = self.atraso_atual[param].get(val, 0)
                intervalo = self.intervalo_medio[param].get(val, len(self.contests))
                if intervalo > 0:
                    score += (atraso / intervalo - 1.0)  # >0 se atrasado
        return score / len(STRUCTURAL_PARAMS)

    # Camada 4: Correlação condicional
    def score_correlacao(self, game_features):
        """
        Exemplo: dado que moldura = 10, qual a distribuição típica de pares?
        Penaliza combinações atípicas.
        """
        score = 0.0
        # Pares de parâmetros correlacionados
        pairs = [('moldura', 'pares'), ('primos', 'pares'), ('repeticoes', 'moldura')]
        for p1, p2 in pairs:
            if p1 in game_features and p2 in game_features:
                v1, v2 = game_features[p1], game_features[p2]
                # Frequência conjunta nos últimos 200 concursos
                recent = self.contests[-200:] if len(self.contests) >= 200 else self.contests
                joint_count = 0
                total_with_v1 = 0
                for i, c in enumerate(recent):
                    prev = self.contests[len(self.contests)-len(recent)+i-1]['dezenas'] if i > 0 else None
                    f1 = extract_filter(c['dezenas'], p1, prev)
                    f2 = extract_filter(c['dezenas'], p2, prev)
                    if f1 == v1:
                        total_with_v1 += 1
                        if f2 == v2:
                            joint_count += 1
                if total_with_v1 > 0:
                    prob = joint_count / total_with_v1
                    # Penaliza se a combinação é rara (prob < 0.2)
                    if prob < 0.2:
                        score += (0.2 - prob) * 5.0
        return score

    # Camada 5: Matriz de transição (Markov)
    def score_transicao(self, game_features):
        """
        Dado o último estado de cada parâmetro, qual a probabilidade do estado do jogo?
        Penaliza transições improváveis.
        """
        if len(self.contests) < 2:
            return 0.0
        score = 0.0
        for param in STRUCTURAL_PARAMS:
            if param in game_features:
                s = self.series[param]
                if len(s) >= 2:
                    prev_val = s[-1]
                    curr_val = game_features[param]
                    # Contar transições prev_val → curr_val
                    trans_count = 0
                    total_prev = 0
                    for i in range(1, len(s)):
                        if s[i-1] == prev_val:
                            total_prev += 1
                            if s[i] == curr_val:
                                trans_count += 1
                    if total_prev > 0:
                        prob = trans_count / total_prev
                        # Penaliza se a transição é improvável (< 0.15)
                        if prob < 0.15:
                            score += (0.15 - prob) * 5.0
        return score

    # Score global
    def compute_global_score(self, game_features, centers, weights):
        """
        Score combinado das 5 camadas.
        Pesos: centro 30%, persistência 25%, ciclo 25%, correlação 10%, transição 10%
        """
        s_centro = self.score_centro(game_features, centers, weights)
        s_persist = self.score_persistencia(game_features)
        s_ciclo = self.score_ciclo(game_features)
        s_correl = self.score_correlacao(game_features)
        s_trans = self.score_transicao(game_features)
        return (0.30 * s_centro + 0.25 * s_persist + 0.25 * s_ciclo +
                0.10 * s_correl + 0.10 * s_trans)

# ============================================================
# STRUCTURAL PREDICTOR (CENTROS E PESOS)
# ============================================================
class StructuralPredictorV72:
    def __init__(self, contests, cache_file='v72_weights_cache.json'):
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
# OTIMIZADOR DE CARTEIRA v72 (COM MODELO DE PRESSÃO)
# ============================================================
class PortfolioOptimizerV72:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None,
                 use_pressure_model=False):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.predictor = StructuralPredictorV72(contests)
        self.use_pressure_model = use_pressure_model
        if use_pressure_model:
            self.pressure = PressaoEstruturalTemporal(contests)
        else:
            self.pressure = None

    def _compute_game_features(self, game, prev_dezenas):
        return {
            'pares': extract_filter(game, 'pares'),
            'moldura': extract_filter(game, 'moldura'),
            'primos': extract_filter(game, 'primos'),
            'amplitude': extract_filter(game, 'amplitude'),
            'repeticoes': extract_filter(game, 'repeticoes', prev_dezenas)
        }

    def _score_game(self, game, centers, prev_dezenas):
        features = self._compute_game_features(game, prev_dezenas)
        if self.use_pressure_model and self.pressure is not None:
            return self.pressure.compute_global_score(features, centers, self.predictor.weights)
        else:
            # Score plano (v71): apenas distância ao centro
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
# WALK‑FORWARD COMPARATIVO: PLANO vs PRESSÃO
# ============================================================
def walk_forward_compare(contests, train_size=500, step=50):
    print(f"\n🔬 WALK‑FORWARD: MODELO PLANO vs MODELO DE PRESSÃO")
    print(f"   Treino: {train_size} | Teste: {step}\n")
    results = {'plano': [], 'pressao': []}
    start = train_size
    while start + step <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+step]
        for modelo, use_press in [('plano', False), ('pressao', True)]:
            try:
                opt = PortfolioOptimizerV72(train_data, use_pressure_model=use_press)
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
    # Consolidação
    print(f"\n📊 RESULTADO FINAL:")
    for modelo in ['plano', 'pressao']:
        avg_lift = np.mean([r['lift'] for r in results[modelo]]) if results[modelo] else 0
        avg_roi = np.mean([r['roi'] for r in results[modelo]]) if results[modelo] else 0
        total_13 = sum(r['13pts'] for r in results[modelo])
        total_14 = sum(r['14pts'] for r in results[modelo])
        nome = "Modelo plano (v71)" if modelo == 'plano' else "Modelo de pressão (v72)"
        print(f"   {nome}:")
        print(f"      Média lift: {avg_lift:.3f} | Média ROI: {avg_roi:.1f}%")
        print(f"      Total 13pts: {total_13} | 14pts: {total_14}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v72")
    print("   MODELO DE PRESSÃO ESTRUTURAL TEMPORAL")
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
        print("3. Exibir pressão atual dos parâmetros estruturais")
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
            opt = PortfolioOptimizerV72(contests, fixed=fixed, semifixed=semifixed,
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
            pressure = PressaoEstruturalTemporal(contests)
            print(f"\n📊 PRESSÃO ATUAL DOS PARÂMETROS ESTRUTURAIS:")
            for param in STRUCTURAL_PARAMS:
                s = pressure.series[param]
                curr_val = s[-1]
                atraso = pressure.atraso_atual[param].get(curr_val, 0)
                intervalo = pressure.intervalo_medio[param].get(curr_val, 1)
                pers_atual = pressure.persistencia_atual[param].get(curr_val, 1)
                pers_media = pressure.persistencia_media[param].get(curr_val, 1.0)
                ciclo_score = atraso / intervalo if intervalo > 0 else 0
                persist_score = pers_atual / pers_media if pers_media > 0 else 1.0
                print(f"   {param:<12}: valor atual={curr_val}, atraso={atraso}, "
                      f"ciclo={ciclo_score:.2f}, persistência={persist_score:.2f}")

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
