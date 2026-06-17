#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v69.1
CORREÇÕES + TESTE DE REVERSÃO DAS DEZENAS

CORREÇÕES:
✅ Bug corrigido: variável 'longa' → 'long_data' (compute_scores)
✅ Normalização dos scores de dezenas por z-score (preserva sinal)
✅ Soma removida dos filtros ativos (peso quase nulo)
✅ Novo módulo (opção 5): Teste de reversão das dezenas
   - Divide dezenas em quintis por freq20 - freq100
   - Mede frequência nos próximos N concursos
   - Verifica se hipótese de reversão se confirma
"""

import numpy as np
from scipy.stats import hypergeom
from collections import Counter
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
# STRUCTURAL PREDICTOR COM CACHE (SOMA REMOVIDA)
# ============================================================
class StructuralPredictorV691:
    def __init__(self, contests, cache_file='v69_weights_cache.json'):
        self.contests = contests
        # Soma removida dos filtros ativos
        self.active_filters = ['pares', 'moldura', 'primos', 'amplitude', 'repeticoes']
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
        """Carrega pesos do cache ou calcula e salva."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                self.weights = data['weights']
                self.gains = data['gains']
                print(f"📂 Pesos carregados do cache ({self.cache_file})")
                return
            except:
                pass
        print("📊 Calculando ganhos reais dos filtros (walk‑forward)...")
        self._compute_real_gains()
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'weights': self.weights, 'gains': self.gains}, f)
            print(f"   Cache salvo em {self.cache_file}")
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
                centers[filtro] = np.median(preds)
        return centers

# ============================================================
# MODELO DE DEZENAS POR REVERSÃO (CORRIGIDO)
# ============================================================
class DezenaModelReversao:
    def __init__(self, contests):
        self.contests = contests

    def compute_scores(self, janela_curta=20, janela_longa=100, peso_atraso=0.3):
        """
        Score baseado em reversão (z‑score):
        - freq20 - freq100 > 0 → penaliza
        - freq20 - freq100 < 0 → bonifica
        - Bônus adicional para dezenas atrasadas
        Retorna scores com média 0 e desvio 1.
        """
        if len(self.contests) == 0:
            return {d: 0.0 for d in range(1, 26)}
        total = len(self.contests)
        recent = self.contests[-janela_curta:] if janela_curta < total else self.contests
        long_data = self.contests[-janela_longa:] if janela_longa < total else self.contests
        
        # Frequências
        freq_curta = Counter()
        for c in recent:
            freq_curta.update(c['dezenas'])
        freq_longa = Counter()
        for c in long_data:
            freq_longa.update(c['dezenas'])
        
        # Atraso
        last_seen = {d: -1 for d in range(1, 26)}
        for i, c in enumerate(self.contests):
            for d in c['dezenas']:
                last_seen[d] = i
        atraso = {d: (total - 1 - last_seen[d]) for d in range(1, 26)}
        max_atraso = max(atraso.values()) + 1
        
        scores = {}
        n_curta = len(recent)
        n_longa = max(1, len(long_data))  # proteção contra divisão por zero
        
        for d in range(1, 26):
            fc = freq_curta.get(d, 0) / n_curta
            fl = freq_longa.get(d, 0) / n_longa  # CORRIGIDO: era 'longa'
            edge = fc - fl  # positivo = quente, negativo = fria
            # Reversão: penaliza quentes, bonifica frias
            reversao_score = -edge * 2.0
            # Bônus por atraso (normalizado)
            atraso_score = peso_atraso * (atraso[d] / max_atraso)
            scores[d] = reversao_score + atraso_score
        
        # Normalização z‑score (preserva sinal e magnitude relativa)
        mean_score = np.mean(list(scores.values()))
        std_score = np.std(list(scores.values()))
        if std_score > 0:
            for d in scores:
                scores[d] = (scores[d] - mean_score) / std_score
        else:
            for d in scores:
                scores[d] = 0.0
        
        return scores

# ============================================================
# TESTE DE REVERSÃO DAS DEZENAS (NOVO)
# ============================================================
def testar_reversao_dezenas(contests, janela_curta=20, janela_longa=100, 
                            n_quintis=5, n_proximos=50, min_history=500):
    """
    Testa se a hipótese de reversão se confirma:
    - Divide as dezenas em quintis baseados em freq20 - freq100
    - Mede a frequência média nos próximos n_proximos concursos
    - Verifica se as frias (Q1) aparecem mais que as quentes (Q5)
    """
    print(f"\n🧪 TESTE DE REVERSÃO DAS DEZENAS")
    print(f"   Janela curta: {janela_curta} | Janela longa: {janela_longa}")
    print(f"   Quintis: {n_quintis} | Próximos concursos: {n_proximos}")
    print(f"   Histórico mínimo: {min_history}\n")
    
    resultados_por_quintil = {q: [] for q in range(1, n_quintis+1)}
    total_testes = 0
    
    for t in tqdm(range(min_history, len(contests) - n_proximos), desc="Testando reversão"):
        treino = contests[:t]
        teste = contests[t:t+n_proximos]
        
        if len(treino) < janela_longa:
            continue
        
        recent = treino[-janela_curta:]
        long_data = treino[-janela_longa:]
        
        freq_curta = Counter()
        for c in recent:
            freq_curta.update(c['dezenas'])
        freq_longa = Counter()
        for c in long_data:
            freq_longa.update(c['dezenas'])
        
        # Calcular edge para cada dezena
        edges = {}
        n_curta = len(recent)
        n_longa = max(1, len(long_data))
        for d in range(1, 26):
            fc = freq_curta.get(d, 0) / n_curta
            fl = freq_longa.get(d, 0) / n_longa
            edges[d] = fc - fl
        
        # Ordenar dezenas por edge
        sorted_dezenas = sorted(edges.items(), key=lambda x: x[1])
        
        # Dividir em quintis
        dezenas_por_quintil = {q: [] for q in range(1, n_quintis+1)}
        for i, (d, _) in enumerate(sorted_dezenas):
            quintil = min(n_quintis, (i * n_quintis) // 25 + 1)
            dezenas_por_quintil[quintil].append(d)
        
        # Medir frequência nos próximos concursos
        freq_teste = Counter()
        for c in teste:
            freq_teste.update(c['dezenas'])
        
        for q in range(1, n_quintis+1):
            dezenas_q = dezenas_por_quintil[q]
            freq_media = np.mean([freq_teste.get(d, 0) / len(teste) for d in dezenas_q])
            resultados_por_quintil[q].append(freq_media)
        
        total_testes += 1
    
    # Resultados agregados
    print(f"   Total de testes: {total_testes}")
    print(f"\n📊 FREQUÊNCIA MÉDIA NOS PRÓXIMOS {n_proximos} CONCURSOS POR QUINTIL:")
    print(f"   {'Quintil':<10} {'Descrição':<20} {'Freq Média':<12} {'Desvio':<10}")
    print(f"   {'-'*50}")
    
    medias_por_quintil = {}
    for q in range(1, n_quintis+1):
        if resultados_por_quintil[q]:
            media = np.mean(resultados_por_quintil[q])
            std = np.std(resultados_por_quintil[q])
            medias_por_quintil[q] = media
            if q == 1:
                desc = "Mais frias"
            elif q == n_quintis:
                desc = "Mais quentes"
            else:
                desc = f"Intermediárias {q}"
            print(f"   Q{q:<9} {desc:<20} {media*100:<12.2f}% {std*100:<10.2f}%")
    
    # Verificar se Q1 (frias) > Q5 (quentes)
    if 1 in medias_por_quintil and n_quintis in medias_por_quintil:
        dif = medias_por_quintil[1] - medias_por_quintil[n_quintis]
        print(f"\n📊 DIFERENÇA Q1 - Q{n_quintis}: {dif*100:+.2f}%")
        if dif > 0.02:
            print(f"   🔍 Evidência de reversão: dezenas frias aparecem mais que quentes.")
        elif dif < -0.02:
            print(f"   📊 Evidência de tendência: dezenas quentes continuam aparecendo mais.")
        else:
            print(f"   ✅ Sem diferença significativa: hipótese de reversão não confirmada.")
    
    return resultados_por_quintil

# ============================================================
# KMEANS MANUAL
# ============================================================
class SimpleKMeans:
    def __init__(self, n_clusters=10, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit_predict(self, X):
        n = len(X)
        if n <= self.n_clusters:
            return np.arange(n)
        idx = np.random.choice(n, self.n_clusters, replace=False)
        centroids = X[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(self.max_iter):
            for i in range(n):
                dists = np.sum((centroids - X[i])**2, axis=1)
                labels[i] = np.argmin(dists)
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
# OTIMIZADOR DE CARTEIRA v69.1
# ============================================================
class PortfolioOptimizerV691:
    def __init__(self, contests, fixed=None, semifixed=None, min_semifixed=0, max_semifixed=None):
        self.contests = contests
        self.generator = LooseGenerator()
        self.fixed = fixed if fixed else []
        self.semifixed = semifixed if semifixed else []
        self.min_semifixed = min_semifixed
        self.max_semifixed = max_semifixed
        self.predictor = StructuralPredictorV691(contests)
        self.dezena_model = DezenaModelReversao(contests)

    def _compute_game_features(self, game, prev_dezenas):
        return {
            'pares': extract_filter(game, 'pares'),
            'moldura': extract_filter(game, 'moldura'),
            'primos': extract_filter(game, 'primos'),
            'amplitude': extract_filter(game, 'amplitude'),
            'repeticoes': extract_filter(game, 'repeticoes', prev_dezenas)
        }

    def _score_game(self, game, centers, prev_dezenas, dezena_scores):
        features = self._compute_game_features(game, prev_dezenas)
        score_estrutural = 0.0
        for filtro, center in centers.items():
            if filtro in features and filtro in self.predictor.weights:
                if filtro == 'amplitude':
                    dist = abs(features[filtro] - center) / 14.0
                else:
                    dist = abs(features[filtro] - center)
                score_estrutural += self.predictor.weights[filtro] * dist
        # Bônus das dezenas (z‑score: valores negativos penalizam, positivos bonificam)
        bonus_dezenas = sum(dezena_scores.get(d, 0.0) for d in game)
        return score_estrutural - 0.3 * bonus_dezenas

    def _build_feature_matrix(self, games, prev_dezenas):
        features = []
        for g in games:
            f = self._compute_game_features(g, prev_dezenas)
            vec = [f['pares'], f['moldura'], f['primos'], f['amplitude']/24, f['repeticoes']]
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
        print(f"\n🧩 CARTEIRA v69.1: {n_games} jogos")
        if self.fixed: print(f"   Fixas: {self.fixed}")
        if self.semifixed:
            print(f"   Semifixas: {self.semifixed} (mín={self.min_semifixed}, máx={self.max_semifixed})")
        centers = self.predictor.predict_centers()
        dezena_scores = self.dezena_model.compute_scores()
        print(f"\n📊 CENTROS PREVISTOS:")
        for f in centers:
            print(f"   {f:<12}: {centers[f]:<6} (peso={self.predictor.weights[f]:.3f})")
        top5 = sorted(dezena_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        bottom5 = sorted(dezena_scores.items(), key=lambda x: x[1])[:5]
        print(f"📊 TOP 5 DEZENAS (reversão): {top5}")
        print(f"📊 BOTTOM 5 DEZENAS (quentes): {bottom5}")
        t0 = time.time()
        prev_dezenas = self.contests[-1]['dezenas'] if self.contests else None
        pool = self.generate_pool(n_candidates, prev_dezenas)
        print(f"   Pool gerado: {len(pool)} jogos")
        if len(pool) < n_games:
            raise RuntimeError(f"Pool insuficiente: {len(pool)} < {n_games}.")
        scored = [(self._score_game(g, centers, prev_dezenas, dezena_scores), g) for g in pool]
        scored.sort(key=lambda x: x[0])
        top_pool = [g for _, g in scored[:2000]]
        X = self._build_feature_matrix(top_pool, prev_dezenas)
        kmeans = SimpleKMeans(n_clusters=min(n_clusters, len(top_pool)))
        labels = kmeans.fit_predict(X)
        cluster_best = {i: [] for i in range(n_clusters)}
        for i, (s, g) in enumerate(scored[:2000]):
            lbl = labels[i]
            cluster_best[lbl].append((s, g))
        diverse_pool = []
        for lbl in cluster_best:
            cluster_best[lbl].sort(key=lambda x: x[0])
            diverse_pool.extend([g for _, g in cluster_best[lbl][:top_per_cluster]])
        print(f"   Pool diversificado: {len(diverse_pool)} jogos (de {n_clusters} clusters)")
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
# WALK‑FORWARD REAL DA CARTEIRA
# ============================================================
def walk_forward_carteira(contests, n_games=5, train_size=500, step=50):
    print(f"\n🔬 WALK‑FORWARD REAL DA CARTEIRA")
    print(f"   Treino: {train_size} | Teste: {step} | Jogos: {n_games}\n")
    results = []
    start = train_size
    while start + step <= len(contests):
        train_data = contests[start-train_size:start]
        test_data = contests[start:start+step]
        try:
            opt = PortfolioOptimizerV691(train_data)
            portfolio = opt.optimize(n_games, 30000, n_clusters=20, top_per_cluster=3)
            bt = opt.backtest(portfolio, test_data)
            results.append({
                'start': start,
                'lift': bt['lift'],
                'roi': bt['roi'],
                '13pts': bt['hit_distribution'].get(13, 0),
                '14pts': bt['hit_distribution'].get(14, 0),
                '15pts': bt['hit_distribution'].get(15, 0),
            })
            print(f"   Janela {start}: lift={bt['lift']:.3f} | ROI={bt['roi']:+.1f}% | "
                  f"13pts={bt['hit_distribution'].get(13,0)} 14pts={bt['hit_distribution'].get(14,0)}")
        except Exception as e:
            print(f"   Janela {start}: ERRO - {e}")
        start += step
    if results:
        print(f"\n📊 RESUMO:")
        print(f"   Média lift: {np.mean([r['lift'] for r in results]):.3f}")
        print(f"   Média ROI: {np.mean([r['roi'] for r in results]):.1f}%")
        print(f"   Total 13pts: {sum(r['13pts'] for r in results)}")
        print(f"   Total 14pts: {sum(r['14pts'] for r in results)}")
        print(f"   Total 15pts: {sum(r['15pts'] for r in results)}")
    return results

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    print("="*70)
    print("🔬 LABORATÓRIO DE ANÁLISE ESTRUTURAL DA LOTOFÁCIL – v69.1")
    print("   CORREÇÕES + TESTE DE REVERSÃO DAS DEZENAS")
    print("="*70)
    contests = load_all_contests('resultados_lotofacil.csv')
    if not contests:
        print("❌ Arquivo 'resultados_lotofacil.csv' não encontrado.")
        return
    print(f"\n📂 {len(contests)} concursos")
    print(f"📌 Último: {contests[-1]['concurso']} - {contests[-1]['dezenas']}")

    while True:
        print("\nOpções:")
        print("1. Gerar carteira otimizada (v69.1)")
        print("2. Walk‑forward real da carteira")
        print("3. Backtest nos últimos 200 concursos")
        print("4. Mostrar centros previstos e scores de dezenas")
        print("5. Teste de reversão das dezenas")
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
            opt = PortfolioOptimizerV691(contests, fixed=fixed, semifixed=semifixed,
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
            walk_forward_carteira(contests)

        elif op == '3':
            fixed_str = input("\n   Fixas (ENTER para pular): ").strip()
            fixed = [int(x) for x in fixed_str.split()] if fixed_str else []
            opt = PortfolioOptimizerV691(contests, fixed=fixed)
            portfolio = opt.optimize(5, 50000, n_clusters=20, top_per_cluster=3)
            bt = opt.backtest(portfolio, contests[-200:])
            print(f"\n🔬 BACKTEST (200): Lift={bt['lift']:.2f}x | ROI={bt['roi']:+.1f}%")
            print(f"   11={bt['hit_distribution'].get(11,0)} 12={bt['hit_distribution'].get(12,0)} "
                  f"13={bt['hit_distribution'].get(13,0)} 14={bt['hit_distribution'].get(14,0)}")

        elif op == '4':
            predictor = StructuralPredictorV691(contests)
            centers = predictor.predict_centers()
            dezena_model = DezenaModelReversao(contests)
            scores = dezena_model.compute_scores()
            print(f"\n📊 CENTROS PREVISTOS E PESOS:")
            for f in predictor.active_filters:
                print(f"   {f:<12}: centro={centers[f]:<6} peso={predictor.weights[f]:.3f}")
            print(f"\n📊 SCORES DE DEZENAS (z‑score, reversão):")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for d, s in sorted_scores:
                bar = "█" * int(max(0, s * 5)) if s > 0 else "█" * int(max(0, -s * 5))
                print(f"   {d:2d}: {s:+6.3f} {bar}")

        elif op == '5':
            testar_reversao_dezenas(contests)

        elif op == '0':
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    main()
