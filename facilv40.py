#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v68
PESOS REAIS + MODELO DE DEZENAS + CLUSTERIZAÇÃO ESTRUTURAL

MELHORIAS:
✅ Pesos dos filtros = ganho real no walk‑forward (taxa predictor − baseline)
✅ Modelo de dezenas individuais: score por frequência recente + atraso + repetição
✅ Clusterização estrutural (KMeans) para forçar diversidade
✅ Score final = distância estrutural ponderada − bônus das dezenas
✅ Mantém pair covering e backtest
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter
from itertools import combinations
import os, random, time, warnings
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
# GERADOR DE JOGOS (SEM FILTROS ESTRUTURAIS)
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
# FUNÇÕES AUXILIARES PARA EXTRAÇÃO DE FILTROS
# ============================================================
def extract_filter(dezenas, filter_name, prev_dezenas=None):
    d = sorted(dezenas)
    if filter_name == 'pares': return sum(1 for x in d if x % 2 == 0)
    if filter_name == 'moldura': return sum(1 for x in d if x in MOLDURA)
    if filter_name == 'primos': return sum(1 for x in d if x in PRIMES)
    if filter_name == 'soma': return sum(d)
    if filter_name == 'amplitude': return max(d) - min(d)
    if filter_name == 'repeticoes':
        if prev_dezenas is None: return 8
        return len(set(d) & set(prev_dezenas))
    return 0

# ============================================================
# STRUCTURAL PREDICTOR + CÁLCULO DE GANHOS REAIS
# ============================================================
class StructuralPredictorV68:
    def __init__(self, contests):
        self.contests = contests
        self.active_filters = ['pares', 'moldura', 'primos', 'soma', 'amplitude', 'repeticoes']
        self.windows = [20, 50, 100, 200]
        self._compute_real_gains()

    def _get_filter_series(self, filter_name):
        series = []
        for i, c in enumerate(self.contests):
            prev = self.contests[i-1]['dezenas'] if i > 0 else None
            series.append(extract_filter(c['dezenas'], filter_name, prev))
        return np.array(series, dtype=float)

    def _compute_real_gains(self, min_history=500):
        """Calcula o ganho real de cada filtro sobre o baseline ingênuo."""
        print("📊 Calculando ganhos reais dos filtros (walk‑forward)...")
        gains = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            acertos_pred, acertos_base = 0, 0
            total = 0
            for t in range(min_history, len(self.contests) - 1):
                history = series[:t+1]
                next_val = series[t+1]
                curr_val = series[t]
                # Predictor: mediana do ensemble de janelas
                preds = []
                for w in self.windows:
                    if len(history) >= w:
                        preds.append(np.median(history[-w:]))
                if preds:
                    center = np.median(preds)
                    if filtro == 'soma':
                        center = round(center / 5) * 5
                    if int(round(center)) == int(next_val):
                        acertos_pred += 1
                # Baseline: repetir valor atual
                if int(curr_val) == int(next_val):
                    acertos_base += 1
                total += 1
            if total > 0:
                gains[filtro] = (acertos_pred / total - acertos_base / total) * 100
            else:
                gains[filtro] = 0.0
        # Normalizar para soma = 1 (apenas ganhos positivos)
        total_gain = sum(max(0, g) for g in gains.values())
        if total_gain > 0:
            self.weights = {f: max(0, gains[f]) / total_gain for f in gains}
        else:
            self.weights = {f: 1.0/len(gains) for f in gains}
        print("   Pesos calculados:")
        for f in self.active_filters:
            print(f"   {f:<12}: ganho={gains[f]:+.2f}% → peso={self.weights[f]:.3f}")

    def predict_centers(self):
        centers = {}
        for filtro in self.active_filters:
            series = self._get_filter_series(filtro)
            preds = []
            for w in self.windows:
                if len(series) >= w:
                    preds.append(np.median(series[-w:]))
            if preds:
                center = np.median(preds)
                if filtro == 'soma':
                    center = round(center / 5) * 5
                centers[filtro] = center
        return centers

# ============================================================
# MODELO DE DEZENAS INDIVIDUAIS
# ============================================================
class DezenaModel:
    def __init__(self, contests):
        self.contests = contests

    def compute_scores(self, janela_freq=50, peso_atraso=0.5, peso_rep=1.0):
        """Retorna dicionário com score para cada dezena (1 a 25)."""
        if len(self.contests) == 0:
            return {d: 0.0 for d in range(1, 26)}
        total = len(self.contests)
        recent = self.contests[-janela_freq:] if janela_freq < total else self.contests
        # Frequência recente
        freq = Counter()
        for c in recent:
            freq.update(c['dezenas'])
        # Atraso
        last_seen = {d: -1 for d in range(1, 26)}
        for i, c in enumerate(self.contests):
            for d in c['dezenas']:
                last_seen[d] = i
        atraso = {d: (total - 1 - last_seen[d]) for d in range(1, 26)}
        max_atraso = max(atraso.values()) + 1
        # Repetição do último concurso
        ultimo = set(self.contests[-1]['dezenas']) if total > 0 else set()
        scores = {}
        for d in range(1, 26):
            f = freq.get(d, 0) / len(recent)
            a = 1.0 / (atraso[d] + 1)
            r = 1.0 if d in ultimo else 0.0
            scores[d] = f + peso_atraso * a + peso_rep * r
        # Normalizar
        total_score = sum(scores.values())
        if total_score > 0:
            for d in scores:
                scores[d] /= total_score
        return scores

# ============================================================
# KMEANS MANUAL (PARA CLUSTERIZAÇÃO ESTRUTURAL)
# ============================================================
class SimpleKMeans:
    def __init__(self, n_clusters=10, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit_predict(self, X):
        n = len(X)
        if n <= self.n_clusters:
            return np.arange(n)
        # Inicializar centróides aleatoriamente
        idx = np.random.choice(n, self.n_clusters, replace=False)
        centroids = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(self.max_iter):
            # Atribuir labels
            for i in range(n):
                dists = np.sum((centroids - X[i])**2, axis=1)
                labels[i] = np.argmin(dists)
            # Atualizar centróides
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    new_centroids[k] = np.mean(members, axis=0)
                else:
                    new_centroids[k] = centroids[k]
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        return labels

# ============================================================
# OTIMIZADOR DE CARTEIRA v68
# ============================================================
class PortfolioOptimizerV68:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.predictor = StructuralPredictorV68(contests)
        self.dezena_model = DezenaModel(contests)

    def _compute_game_features(self, game, prev_dezenas):
        return {
            'pares': extract_filter(game, 'pares'),
            'moldura': extract_filter(game, 'moldura'),
            'primos': extract_filter(game, 'primos'),
            'soma': extract_filter(game, 'soma'),
            'amplitude': extract_filter(game, 'amplitude'),
            'repeticoes': extract_filter(game, 'repeticoes', prev_dezenas)
        }

    def _score_game(self, game, centers, prev_dezenas, dezena_scores):
        """Score combinado: distância estrutural ponderada − bônus das dezenas."""
        features = self._compute_game_features(game, prev_dezenas)
        score_estrutural = 0.0
        for filtro, center in centers.items():
            if filtro in features and filtro in self.predictor.weights:
                if filtro == 'soma':
                    dist = abs(features[filtro] - center) / 150.0
                elif filtro == 'amplitude':
                    dist = abs(features[filtro] - center) / 14.0
                else:
                    dist = abs(features[filtro] - center)
                score_estrutural += self.predictor.weights[filtro] * dist
        # Bônus das dezenas: soma dos scores das 15 dezenas do jogo
        bonus_dezenas = sum(dezena_scores.get(d, 0.0) for d in game)
        return score_estrutural - 0.3 * bonus_dezenas  # peso 0.3 para dezenas

    def _build_feature_matrix(self, games, prev_dezenas):
        """Constrói matriz de features estruturais para clusterização."""
        features = []
        for g in games:
            f = self._compute_game_features(g, prev_dezenas)
            vec = [f['pares'], f['moldura'], f['primos'], f['soma']/200, f['amplitude']/24, f['repeticoes']]
            features.append(vec)
        return np.array(features)

    def generate_pool(self, n_candidates, prev_dezenas=None):
        pool, seen = [], set()
        for _ in tqdm(range(n_candidates), desc="Gerando pool"):
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

    def select_pair_covering(self, candidates, n_select):
        if len(candidates) < n_select:
            return candidates
        covered, selected = set(), []
        for _ in range(n_select):
            best_idx, best_new = -1, -1
            for i, c in enumerate(candidates):
                if c in selected: continue
                pairs = set(combinations(sorted(c), 2))
                new_pairs = len(pairs - covered)
                if new_pairs > best_new:
                    best_new, best_idx = new_pairs, i
            if best_idx == -1: break
            selected.append(candidates[best_idx])
            covered.update(combinations(sorted(candidates[best_idx]), 2))
        return selected

    def optimize(self, n_games=5, n_candidates=50000, n_clusters=20, top_per_cluster=3):
        print(f"\n🧩 CARTEIRA v68: {n_games} jogos")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.semifixed:
            print(f"   Semifixas: {self.semifixed} (mín={self.min_semifixed}, máx={self.max_semifixed})")
        print(f"   Clusters: {n_clusters} | Top por cluster: {top_per_cluster}")

        # Previsões
        centers = self.predictor.predict_centers()
        dezena_scores = self.dezena_model.compute_scores()
        print(f"\n📊 CENTROS PREVISTOS:")
        for f in centers:
            print(f"   {f:<12}: {centers[f]:<6} (peso={self.predictor.weights[f]:.3f})")
        print(f"📊 TOP 5 DEZENAS: {sorted(dezena_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")

        # Gerar pool
        t0 = time.time()
        prev_dezenas = self.contests[-1]['dezenas'] if self.contests else None
        pool = self.generate_pool(n_candidates, prev_dezenas)
        print(f"   Pool gerado: {len(pool)} jogos")

        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")

        # Calcular scores para ordenação inicial
        scored = []
        for g in pool:
            s = self._score_game(g, centers, prev_dezenas, dezena_scores)
            scored.append((s, g))
        scored.sort(key=lambda x: x[0])
        # Pegar os top 2000 para clusterizar
        top_pool = [g for _, g in scored[:2000]]

        # Clusterização estrutural
        X = self._build_feature_matrix(top_pool, prev_dezenas)
        kmeans = SimpleKMeans(n_clusters=min(n_clusters, len(top_pool)))
        labels = kmeans.fit_predict(X)

        # Selecionar os melhores de cada cluster
        cluster_best = {i: [] for i in range(n_clusters)}
        for i, (s, g) in enumerate(scored[:2000]):
            lbl = labels[i]
            cluster_best[lbl].append((s, g))

        diverse_pool = []
        for lbl in cluster_best:
            cluster_best[lbl].sort(key=lambda x: x[0])
            diverse_pool.extend([g for _, g in cluster_best[lbl][:top_per_cluster]])

        print(f"   Pool diversificado: {len(diverse_pool)} jogos (de {n_clusters} clusters)")

        # Pair covering final
        portfolio = self.select_pair_covering(diverse_pool, n_games)
        print(f"✅ Carteira final: {len(portfolio)} jogos em {time.time()-t0:.1f}s")
        return portfolio

    def backtest(self, portfolio, test_draws):
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
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v68")
    print("   PESOS REAIS + MODELO DE DEZENAS + CLUSTERIZAÇÃO")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira otimizada (v68)")
        print("2. Backtest nos últimos 200 concursos")
        print("3. Mostrar centros previstos e pesos")
        print("0. Sair")
        op = input("Escolha: ").strip()

        if op == '1':
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
            opt = PortfolioOptimizerV68(contests, fixed=fixed, semifixed=semifixed,
                                        min_semifixed=min_semi, max_semifixed=max_semi)
            portfolio = opt.optimize(5, 50000, n_clusters=20, top_per_cluster=3)
            for i, g in enumerate(portfolio, 1):
                p = sum(1 for x in g if x%2==0); pr = sum(1 for x in g if x in PRIMES); m = sum(1 for x in g if x in MOLDURA)
                rep = len(set(g) & set(contests[-1]['dezenas'])) if contests else 0
                print(f" {i}. {g} | P:{p} Pr:{pr} M:{m} Rep:{rep}")
            if len(contests) > 200:
                bt = opt.backtest(portfolio, contests[-200:])
                print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
                print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                      f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)}")

        elif op == '2':
            fixed_str = input("\n   Fixas (ENTER para pular): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            opt = PortfolioOptimizerV68(contests, fixed=fixed)
            portfolio = opt.optimize(5, 50000, n_clusters=20, top_per_cluster=3)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)}")

        elif op == '3':
            predictor = StructuralPredictorV68(contests)
            centers = predictor.predict_centers()
            print(f"\n📊 CENTROS PREVISTOS E PESOS:")
            for f in predictor.active_filters:
                print(f"   {f:<12}: centro={centers[f]:<6} peso={predictor.weights[f]:.3f}")

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
